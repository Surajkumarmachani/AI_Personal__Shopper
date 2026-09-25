"""
api_keys.py
===========
API key auth for the backend: every client (and the desktop app) gets its own
key, sent on each request as the `X-API-Key` header.

  - Keys are random (`aura_live_` + 32 url-safe chars) and only their SHA-256
    hash is stored. The raw key is shown ONCE, at creation.
  - Storage: SQLite by default (local dev). Set API_KEYS_BACKEND=firestore to
    use Google Cloud Firestore instead (production on Cloud Run, where the
    container's disk is wiped on every restart).
  - Each key has a per-minute rate limit (RATE_LIMIT_PER_MIN, default 60).
  - Every authenticated request bumps the key's request_count / last_used_at.

Manage keys from the command line (run inside files/ with the venv active):

  python api_keys.py create "Acme Corp"      → prints the new key once
  python api_keys.py list
  python api_keys.py revoke 3                → disable key with id 3

Or, with no terminal access to the server, use the admin endpoints from the
browser at <backend-url>/docs (requires ADMIN_TOKEN to be set on the server):

  POST /admin/keys               {"name": "Acme Corp"}  → returns the new key once
  GET  /admin/keys
  POST /admin/keys/{id}/revoke

To manage the production (Firestore) keys from your laptop:

  gcloud auth application-default login
  API_KEYS_BACKEND=firestore GOOGLE_CLOUD_PROJECT=<project-id> python api_keys.py list
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
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

load_dotenv()

BACKEND = os.getenv("API_KEYS_BACKEND", "sqlite").lower()
DB_PATH = Path(os.getenv("API_KEYS_DB", str(Path(__file__).parent / "api_keys.db")))
FIRESTORE_COLLECTION = os.getenv("API_KEYS_COLLECTION", "api_keys")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
# Password for the /admin endpoints. Unset → admin endpoints are disabled.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
KEY_PREFIX = "aura_live_"

# Distinct scheme names, or /docs merges them into one "APIKeyHeader" and the
# Authorize dialog only offers X-API-Key.
_api_key_header = APIKeyHeader(name="X-API-Key", scheme_name="ApiKey", auto_error=False)
_admin_token_header = APIKeyHeader(name="X-Admin-Token", scheme_name="AdminToken", auto_error=False)
_db_lock = threading.Lock()


@dataclass(frozen=True)
class Client:
    id: str
    name: str


# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────
# Both stores expose the same four methods. `id` is the short identifier shown
# by `list` and passed to `revoke`.
def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _SqliteStore:
    def _connect(self) -> sqlite3.Connection:
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

    def add(self, name: str, key_hash: str, preview: str) -> None:
        with _db_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO api_keys (name, key_hash, key_preview, created_at) VALUES (?, ?, ?, ?)",
                (name, key_hash, preview, _now()),
            )

    def all(self) -> list[dict]:
        with _db_lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, name, key_preview, active, created_at, last_used_at, request_count "
                "FROM api_keys ORDER BY id"
            ).fetchall()
        return [dict(r) | {"id": str(r["id"]), "active": bool(r["active"])} for r in rows]

    def revoke(self, key_id: str) -> bool:
        if not key_id.isdigit():
            return False
        with _db_lock, self._connect() as conn:
            cur = conn.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (int(key_id),))
        return cur.rowcount > 0

    def lookup(self, key_hash: str) -> Client | None:
        with _db_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, name FROM api_keys WHERE key_hash = ? AND active = 1",
                (key_hash,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE api_keys SET request_count = request_count + 1, last_used_at = ? WHERE id = ?",
                (_now(), row[0]),
            )
        return Client(id=str(row[0]), name=row[1])


class _FirestoreStore:
    # One document per key, with the key's hash as the document id, so the
    # per-request lookup is a single document read.
    def __init__(self) -> None:
        from google.cloud import firestore  # only needed in production

        self._firestore = firestore
        self._col = firestore.Client().collection(FIRESTORE_COLLECTION)

    def add(self, name: str, key_hash: str, preview: str) -> None:
        self._col.document(key_hash).set({
            "id": secrets.token_hex(3),
            "name": name,
            "key_preview": preview,
            "active": True,
            "created_at": _now(),
            "last_used_at": None,
            "request_count": 0,
        })

    def all(self) -> list[dict]:
        rows = [d.to_dict() for d in self._col.stream()]
        return sorted(rows, key=lambda r: r["created_at"])

    def revoke(self, key_id: str) -> bool:
        docs = list(self._col.where("id", "==", key_id).stream())
        for d in docs:
            d.reference.update({"active": False})
        return bool(docs)

    def lookup(self, key_hash: str) -> Client | None:
        ref = self._col.document(key_hash)
        snap = ref.get()
        if not snap.exists or not snap.get("active"):
            return None
        ref.update({
            "request_count": self._firestore.Increment(1),
            "last_used_at": _now(),
        })
        return Client(id=snap.get("id"), name=snap.get("name"))


_store = None


def _get_store():
    global _store
    if _store is None:
        _store = _FirestoreStore() if BACKEND == "firestore" else _SqliteStore()
    return _store


def create_key(name: str) -> str:
    """Create a key for `name` and return the raw key (not retrievable later)."""
    key = KEY_PREFIX + secrets.token_urlsafe(24)
    _get_store().add(name, _hash(key), key[:14] + "…")
    return key


def list_keys() -> list[dict]:
    return _get_store().all()


def revoke_key(key_id: str) -> bool:
    return _get_store().revoke(key_id)


def _lookup(key: str) -> Client | None:
    """Return the active client owning `key` and record the usage, else None."""
    return _get_store().lookup(_hash(key))


# ─────────────────────────────────────────────
# RATE LIMITING (sliding 60s window, per key)
# ─────────────────────────────────────────────
# In-memory, so it resets on restart and is per-process. Fine for a single
# server; move to Redis if you run several workers/instances.
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _check_rate_limit(client_id: str) -> None:
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


def require_admin(token: str | None = Security(_admin_token_header)) -> None:
    """Allow only requests carrying the server's ADMIN_TOKEN."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=404, detail="Not found.")
    if not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token.")


# ─────────────────────────────────────────────
# ADMIN ENDPOINTS (manage keys from the browser via /docs)
# ─────────────────────────────────────────────
class NewKeyRequest(BaseModel):
    name: str


admin_router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@admin_router.post("/keys")
def admin_create_key(req: NewKeyRequest) -> dict:
    key = create_key(req.name)
    return {"name": req.name, "api_key": key, "note": "Store this key now — it will not be shown again."}


@admin_router.get("/keys")
def admin_list_keys() -> list[dict]:
    return list_keys()


@admin_router.post("/keys/{key_id}/revoke")
def admin_revoke_key(key_id: str) -> dict:
    if not revoke_key(key_id):
        raise HTTPException(status_code=404, detail=f"No key with id {key_id}.")
    return {"revoked": key_id}


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
                f"{r['id']:>6}  {status}  {r['key_preview']:<16} {r['name']:<24} "
                f"requests={r['request_count']:<6} last_used={r['last_used_at'] or '-'}"
            )
        return 0
    if cmd == "revoke" and len(argv) == 2:
        ok = revoke_key(argv[1])
        print(f"Revoked key {argv[1]}." if ok else f"No key with id {argv[1]}.")
        return 0 if ok else 1
    print(usage)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
