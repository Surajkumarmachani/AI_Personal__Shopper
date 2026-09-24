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

Open http://localhost:5173. The frontend expects the backend at `http://127.0.0.1:8000`.

To (re)build the product index, run `python ingest_to_pinecone.py` from `files/` — it writes to Pinecone and regenerates `bm25_model.json`.
