"""
config.py
=========
Central place for all the knobs: model names, the emotion→tone mapping,
the agent's system prompt, and small shared objects used across files.

Edit MODEL CONSTANTS here to swap models without touching any other file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env once, here, before any OpenAI/Pinecone client is created elsewhere.
load_dotenv()

# ─────────────────────────────────────────────
# MODEL CONSTANTS  — swap models here only
# ─────────────────────────────────────────────
# OpenAI GPT-4o via LangChain ChatOpenAI — used as the ReAct agent's LLM.
LLM_MODEL = "gpt-4o"

# OpenAI text-embedding-3-small for product embeddings (cosine similarity).
# MUST match the model you embedded your catalogue with in ingest_to_pinecone.py.
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM   = 1536

# Pinecone index name (same one your ingestion script wrote to)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "shopping-products")

# How many products to SHOW the shopper (and feed the LLM) per search
TOP_K = 4

# Pull a larger candidate pool from Pinecone for RRF fusion, then take
# the best TOP_K. More candidates → better fusion quality.
CANDIDATE_K = 20

# Minimum cosine similarity (0-1) a product must clear to be shown. Vector search
# ALWAYS returns something — even for a query with no good match — so without a
# floor the assistant surfaces irrelevant items (e.g. a foosball scoring pad for
# a "study table" query). That's what looks like "hallucination": the LLM is
# honestly describing junk it was handed. Raise this to be stricter, lower it to
# be more permissive; 0.0 disables filtering entirely. Tune it using the
# per-match scores now logged by search_products. For bge-small + cosine,
# genuinely relevant matches usually land around 0.4-0.7.
MIN_SCORE = 0.30

# ─────────────────────────────────────────────
# HYBRID RETRIEVAL (Dense + Sparse → RRF)
# ─────────────────────────────────────────────
# Reciprocal Rank Fusion smoothing constant. k=60 is the standard value from
# the original Cormack, Clarke & Buettcher paper. Higher k reduces the
# influence of high-ranked items; lower k amplifies them.
RRF_K = 60

# Path to the fitted BM25 model (written by ingest_to_pinecone.py, loaded by
# product_search.py at startup). Lives next to the Python files by default.
BM25_MODEL_PATH = Path(os.getenv(
    "BM25_MODEL_PATH",
    str(Path(__file__).parent / "bm25_model.json"),
))


# ─────────────────────────────────────────────
# EMOTION → TONE MAPPING
# ─────────────────────────────────────────────
# The detected emotion (from webcam, voice, or text) maps to a short tone
# instruction that gets injected into each turn. Add/edit freely.
EMOTION_TONE_MAP = {
    "happy":     "Match their upbeat energy — be warm, enthusiastic, and celebratory.",
    "sad":       "Be gentle, warm, and reassuring. Lean toward comforting or mood-lifting "
                 "suggestions and don't overwhelm them with too many options.",
    "angry":     "Stay calm, patient, and concise. Focus on quickly solving their problem "
                 "and skip the cheerfulness.",
    "surprised": "Be engaged and curious; help them explore and make sense of the options.",
    "fearful":   "Be reassuring and clear. Keep choices simple and low-pressure.",
    "disgusted": "Be matter-of-fact and take their reaction seriously. Pivot quickly to "
                 "better alternatives.",
    "calm":      "Keep a relaxed, friendly, conversational tone.",
    "neutral":   "Be friendly, balanced, and helpful.",
}

EMOTION_EMOJI_MAP = {
    "happy": "😊",
    "sad": "😔",
    "angry": "😠",
    "surprised": "😮",
    "fearful": "😟",
    "disgusted": "🤢",
    "calm": "😌",
    "neutral": "😐",
}


def emotion_to_emoji(emotion: str) -> str:
    emotion = (emotion or "neutral").lower()
    return EMOTION_EMOJI_MAP.get(emotion, "😐")


def wrap_with_emotion(message: str, emotion: str) -> str:
    """
    Prepend a bracketed tone note to the user's message based on the current
    emotion. The agent's system prompt tells it to honor this note but never
    repeat it. For 'neutral' (or unknown) we add nothing — keeps prompts clean.
    """
    emotion = (emotion or "neutral").lower()
    if emotion == "neutral" or emotion not in EMOTION_TONE_MAP:
        return message
    tone = EMOTION_TONE_MAP[emotion]
    return (
        f"[Conversation context — the shopper currently seems {emotion}. {tone}]\n\n"
        f"{message}"
    )


# ─────────────────────────────────────────────
# AGENT SYSTEM PROMPT
# ─────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are Aura, a warm and perceptive AI shopping assistant for an online store.

YOUR JOB
- Help shoppers discover products they'll love through natural conversation.
- Whenever the shopper describes a need, a budget, an occasion, a style, or asks
  for recommendations, you MUST call the `search_products` tool to find real
  items. Never invent products, prices, brands, or details — only recommend
  items the tool actually returns.
- STRICT INSTRUCTION: When calling the `search_products` tool, ensure your JSON arguments are perfectly formatted. Do NOT use single quotes inside JSON strings, and do NOT output raw XML-like function calls in text.
- After searching, present 2-4 of the best matches conversationally. For each,
  give the name, the price, and one short reason it fits what they asked for.
  Keep it skimmable — not a wall of text.
- IMPORTANT: When you successfully find and present products, do NOT ask the
  shopper follow-up questions like "are you looking for anything else?" or
  "do any of these interest you?" or similar. Just present the results and
  stop. Let the shopper drive the conversation. Only speak again when they
  ask something.
- If the tool returns nothing relevant, say so honestly and ask one clarifying
  question to narrow the search.
- For greetings, small talk, or questions that aren't about finding a product,
  just respond naturally without searching.

TONE
- Each user message may begin with a bracketed context note describing the
  shopper's current emotional state and how to adapt your tone. Honor that
  guidance in how you respond, but NEVER repeat, quote, or refer to the note
  itself — the shopper cannot see it.
- Be a friendly, helpful guide, not a pushy salesperson. Stay concise.
"""


# ─────────────────────────────────────────────
# SHARED OBJECTS
# ─────────────────────────────────────────────
# The search tool writes its structured results here, keyed by session_id, so
# the /chat endpoint can return product cards to the frontend. Cleared each turn.
# (Fine for local/dev. For multi-user production, swap this for Redis.)
PRODUCT_BUFFER: dict[str, list[dict]] = {}