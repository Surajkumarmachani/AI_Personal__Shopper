"""
api_keys.py
===========
API key auth for the backend: every client (and the desktop app) gets its own
key, sent on each request as the `X-API-Key` header.

  - Keys are random (`aura_live_` + 32 url-safe chars) and only their SHA-256
    hash is stored, in SQLite. The raw key is shown ONCE, at creation.
  - Each key has a per-minute rate limit (RATE_LIMIT_PER_MIN, default 60).
  - Every authenticated request bumps the key's request_count / last_used_at.

Manage keys from the command line (run inside files/ with the venv active):

  python api_keys.py create "Acme Corp"      → prints the new key once
  python api_keys.py list
  python api_keys.py revoke 3                → disable key with id 3
"""

import hashlib
import os
import secrets
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

load_dotenv()

DB_PATH = Path(os.getenv("API_KEYS_DB", str(Path(__file__).parent / "api_keys.db")))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
KEY_PREFIX = "aura_live_"

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_db_lock = threading.Lock()


@dataclass(frozen=True)
class Client:
    id: int
    name: str


# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            key_hash      TEXT    NOT NULL UNIQUE,
            key_preview   TEXT    NOT NULL,
            active        INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT    NOT NULL,
            last_used_at  TEXT,
            request_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return conn


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_key(name: str) -> str:
    """Create a key for `name` and return the raw key (not retrievable later)."""
    key = KEY_PREFIX + secrets.token_urlsafe(24)
    with _db_lock, _connect() as conn:
        conn.execute(
            "INSERT INTO api_keys (name, key_hash, key_preview, created_at) VALUES (?, ?, ?, ?)",
            (name, _hash(key), key[:14] + "…", _now()),
        )
    return key


def list_keys() -> list[dict]:
    with _db_lock, _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, key_preview, active, created_at, last_used_at, request_count "
            "FROM api_keys ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_key(key_id: int) -> bool:
    with _db_lock, _connect() as conn:
        cur = conn.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,))
    return cur.rowcount > 0


def _lookup(key: str) -> Client | None:
    """Return the active client owning `key` and record the usage, else None."""
    with _db_lock, _connect() as conn:
        row = conn.execute(
            "SELECT id, name FROM api_keys WHERE key_hash = ? AND active = 1",
            (_hash(key),),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE api_keys SET request_count = request_count + 1, last_used_at = ? WHERE id = ?",
            (_now(), row[0]),
        )
    return Client(id=row[0], name=row[1])


# ─────────────────────────────────────────────
# RATE LIMITING (sliding 60s window, per key)
# ─────────────────────────────────────────────
# In-memory, so it resets on restart and is per-process. Fine for a single
# server; move to Redis if you run several workers/instances.
_hits: dict[int, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _check_rate_limit(client_id: int) -> None:
    now = time.monotonic()
    with _hits_lock:
        window = _hits[client_id]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MIN:
            retry_after = int(60 - (now - window[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MIN} requests/minute).",
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)


# ─────────────────────────────────────────────
# FASTAPI DEPENDENCY
# ─────────────────────────────────────────────
def require_api_key(key: str | None = Security(_api_key_header)) -> Client:
    """Reject the request unless it carries a valid, active, non-rate-limited key."""
    if not key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    client = _lookup(key)
    if client is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key.")
    _check_rate_limit(client.id)
    return client


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _cli(argv: list[str]) -> int:
    usage = 'usage: python api_keys.py create "<client name>" | list | revoke <id>'
    if not argv:
        print(usage)
        return 1
    cmd = argv[0]
    if cmd == "create" and len(argv) == 2:
        key = create_key(argv[1])
        print(f"API key for {argv[1]!r} (store it now — it will not be shown again):\n\n  {key}\n")
        return 0
    if cmd == "list" and len(argv) == 1:
        rows = list_keys()
        if not rows:
            print("No API keys yet.")
        for r in rows:
            status = "active " if r["active"] else "REVOKED"
            print(
                f"{r['id']:>3}  {status}  {r['key_preview']:<16} {r['name']:<24} "
                f"requests={r['request_count']:<6} last_used={r['last_used_at'] or '-'}"
            )
        return 0
    if cmd == "revoke" and len(argv) == 2 and argv[1].isdigit():
        ok = revoke_key(int(argv[1]))
        print(f"Revoked key {argv[1]}." if ok else f"No key with id {argv[1]}.")
        return 0 if ok else 1
    print(usage)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
