# AI_Personal__Shopper

Aura — an emotion-aware AI shopping assistant. A FastAPI + LangGraph agent (GPT-4o) does hybrid product search over Pinecone (dense + BM25, fused with RRF), adapting its tone to the shopper's emotion detected from webcam, voice, or text. React + Vite frontend with voice input and TTS replies.

## Structure

- `files/` — Python backend (FastAPI)
- `frontend/` — React + Vite UI

## Setup

**Backend** (Python 3.11)

```bash
cd files
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cp .env.example .env   # then fill in OPENAI_API_KEY, PINECONE_API_KEY, ELEVENLABS_API_KEY, etc.
uvicorn main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Create an API key for the web app and put it in `frontend/.env.local`:

```bash
# in files/, venv active
python api_keys.py create "Local web frontend"
```

```
# frontend/.env.local
VITE_API_BASE=http://127.0.0.1:8000
VITE_API_KEY=aura_live_...
```

Open http://localhost:5173.

## API keys

Every endpoint except `GET /health` requires an `X-API-Key` header. Each client gets its own key; only a hash is stored (SQLite, `files/api_keys.db`).

```bash
python api_keys.py create "Acme Corp"   # prints the key once — send it to the client
python api_keys.py list                 # usage per key
python api_keys.py revoke 3             # disable key id 3
```

- Missing/invalid/revoked key → `401`; over `RATE_LIMIT_PER_MIN` (default 60) → `429` with `Retry-After`.
- Conversation memory is scoped per key, so clients can't see each other's sessions.
- Interactive API docs: `http://127.0.0.1:8000/docs`.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "X-API-Key: aura_live_..." -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message": "running shoes under $100"}'
```

To (re)build the product index, run `python ingest_to_pinecone.py` from `files/` — it writes to Pinecone and regenerates `bm25_model.json`.
