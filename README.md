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

## Deploy

Backend → **Google Cloud Run** (built from this GitHub repo, HTTPS, API keys in Firestore). Frontend → **Vercel**. Everything below is done in the web consoles — no CLI needed.

### 1. Google Cloud project
1. https://console.cloud.google.com → project picker → **New project**. Note the **Project ID**.
2. **Billing** → link a billing account (the free trial credit applies).

### 2. Firestore (stores API keys)
1. Search **Firestore** → **Create database**.
2. Database ID: `(default)` · Edition: Standard · Mode: **Native** · Location: **us-east1** → **Create**.

### 3. Permission for the backend to use Firestore
1. **IAM & Admin → IAM** → find **Compute Engine default service account** (`<number>-compute@developer.gserviceaccount.com`) → ✏️ Edit.
2. **Add another role** → **Cloud Datastore User** → **Save**.

### 4. Deploy the backend on Cloud Run
1. Search **Cloud Run** → **Deploy container** → **Service**.
2. Choose **Continuously deploy from a repository** → **Set up with Cloud Build** → enable the APIs it asks for.
3. Provider **GitHub** → authenticate → repository `AI_Personal__Shopper` → **Next**.
4. Branch `^main$` · Build type **Dockerfile** · Source location `/Dockerfile` (the default) → **Save**.
5. Service name `aura-backend` · Region **us-east1** · Authentication **Allow public access** (our API keys protect it).
6. Billing **Request-based** · Service scaling: minimum instances **1**, maximum instances **1**
   (chat memory and rate limits live in RAM, so exactly one instance; set minimum to 0 to save credit at the cost of ~1 min cold starts).
7. **Containers, volumes, networking, security**:
   - Container port **8080** · Memory **4 GiB** · CPU **2** · Request timeout **300** · ✅ Startup CPU boost.
   - **Variables & secrets** → add each variable from your `files/.env`, plus:
     `API_KEYS_BACKEND=firestore`, `ALLOWED_ORIGINS=*`, `ADMIN_TOKEN=<long random password>`.
8. **Create**. The first build takes ~10–15 min (Cloud Build → History shows progress).
9. Open the service URL + `/health` → `{"status":"ok"}`.

Every `git push` to `main` now rebuilds and redeploys automatically.

### 5. Create API keys (browser)
1. Open `<service-url>/docs`.
2. Click **Authorize** → paste your `ADMIN_TOKEN` into **X-Admin-Token** → Authorize.
3. **POST /admin/keys** → Try it out → `{"name": "Web frontend"}` → Execute → copy `api_key` (shown once).
   Repeat for `"Desktop app"` and each client. **GET /admin/keys** lists usage; **POST /admin/keys/{id}/revoke** disables one.

### 6. Frontend on Vercel
1. https://vercel.com → **Add New → Project** → import this repo.
2. **Root Directory**: `frontend` (preset: Vite).
3. **Environment Variables**: `VITE_API_BASE` = Cloud Run URL, `VITE_API_KEY` = the "Web frontend" key.
4. **Deploy** → `https://<name>.vercel.app`.
5. In Cloud Run → **Edit & deploy new revision** → Variables → set `ALLOWED_ORIGINS=https://<name>.vercel.app` → Deploy.
