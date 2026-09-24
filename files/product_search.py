"""
product_search.py
=================
The LangChain tool the agent calls to find products.

**Hybrid retrieval pipeline (no reranker):**
  Dense top-20  ──────────────┐
                               ├─► RRF fuse ─► top-20 candidates ─► final top-4
  Hybrid (dense+sparse) top-20┘

How it works with Pinecone's dense-type index:
  - Leg 1 (dense-only): query with just the dense vector → pure semantic ranking
  - Leg 2 (hybrid):     query with dense + sparse vectors together → keyword-boosted ranking
  - RRF fuses the two ranked lists by rank position (not raw score)
  - Fused list capped at CANDIDATE_K (20) → TOP_K (4) results returned

When the BM25 model isn't available, Leg 2 is skipped and we fall back to
dense-only retrieval.

Embeddings: OpenAI text-embedding-3-small (via API). Must match the model used
            at ingestion time (ingest_to_pinecone.py).
"""

import logging
import contextvars
from collections import defaultdict
from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain.tools import tool

from config import (
    EMBED_MODEL,
    PINECONE_INDEX_NAME,
    TOP_K,
    CANDIDATE_K,
    MIN_SCORE,
    RRF_K,
    BM25_MODEL_PATH,
    PRODUCT_BUFFER,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Session tracking via contextvars (async-safe)
# ─────────────────────────────────────────────
_session_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="default"
)


def set_session_id(sid: str) -> None:
    """Set the current session_id for the search tool to read."""
    _session_var.set(sid)


# ─────────────────────────────────────────────
# Clients created once at import time
# ─────────────────────────────────────────────
_openai  = OpenAI()                     # reads OPENAI_API_KEY from env
_pc      = Pinecone()                   # reads PINECONE_API_KEY from env
_index   = _pc.Index(PINECONE_INDEX_NAME)

# Load fitted BM25 model for sparse query encoding. If the model file doesn't
# exist yet (haven't re-ingested), we gracefully fall back to dense-only.
_bm25 = None
if BM25_MODEL_PATH.exists():
    try:
        _bm25 = BM25Encoder().load(str(BM25_MODEL_PATH))
        log.info(f"BM25 model loaded from {BM25_MODEL_PATH}")
    except Exception as e:
        log.warning(f"Failed to load BM25 model from {BM25_MODEL_PATH}: {e}")
else:
    log.warning(
        f"BM25 model not found at {BM25_MODEL_PATH}. "
        "Sparse retrieval disabled — run ingest_to_pinecone.py to generate it. "
        "Falling back to dense-only retrieval."
    )


# ─────────────────────────────────────────────
# Embedding helpers (OpenAI API)
# ─────────────────────────────────────────────
def _embed_query(text: str) -> list[float]:
    """Embed the search query using OpenAI text-embedding-3-small."""
    response = _openai.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


def _sparse_encode_query(text: str) -> dict | None:
    """Encode query as a sparse BM25 vector. Returns None if BM25 unavailable."""
    if _bm25 is None:
        return None
    try:
        sparse_vec = _bm25.encode_queries(text)
        # BM25Encoder may return a list of dicts (one per query) or a single dict
        if isinstance(sparse_vec, list):
            return sparse_vec[0] if sparse_vec else None
        return sparse_vec
    except Exception as e:
        log.warning(f"Sparse query encoding failed: {e}")
        return None


# ─────────────────────────────────────────────
# Pinecone match helpers
# ─────────────────────────────────────────────
def _to_dict(match) -> dict:
    """Convert a Pinecone ScoredVector to a plain dict.

    The Pinecone SDK returns ScoredVector objects from queries, not plain dicts.
    These support .get()/.id/.score/.metadata but NOT dict unpacking ({**m}).
    This helper normalizes them so all downstream code can use standard dict ops.
    """
    if isinstance(match, dict):
        return match
    if hasattr(match, "to_dict"):
        return match.to_dict()
    # Fallback: manually build a dict from known attributes
    return {
        "id": getattr(match, "id", ""),
        "score": getattr(match, "score", 0.0),
        "metadata": getattr(match, "metadata", {}),
    }


# ─────────────────────────────────────────────
# RRF fusion
# ─────────────────────────────────────────────
def _rrf_fuse(
    dense_matches: list[dict],
    hybrid_matches: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion: merge two ranked lists by rank position.

    RRF_score(doc) = Σ  1 / (k + rank)   for each list the doc appears in

    This is robust to different score scales because it uses rank, not raw
    score. Deduplicates by document ID automatically.

    Args:
        dense_matches:  results from the dense-only query (pure semantic)
        hybrid_matches: results from the hybrid query (dense + sparse / keyword-boosted)
        k: smoothing constant (default 60, from the original RRF paper)

    Returns a list of match dicts sorted by fused RRF score (highest first).
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    match_data: dict[str, dict] = {}
    source_info: dict[str, set] = defaultdict(set)

    # Score dense-only matches by their rank position
    for rank, match in enumerate(dense_matches, start=1):
        d = _to_dict(match)
        doc_id = d.get("id", "")
        rrf_scores[doc_id] += 1.0 / (k + rank)
        match_data[doc_id] = d
        source_info[doc_id].add("dense")

    # Score hybrid (dense+sparse) matches by their rank position
    for rank, match in enumerate(hybrid_matches, start=1):
        d = _to_dict(match)
        doc_id = d.get("id", "")
        rrf_scores[doc_id] += 1.0 / (k + rank)
        if doc_id not in match_data:
            match_data[doc_id] = d
        source_info[doc_id].add("hybrid")

    # Sort by fused RRF score, descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda doc_id: rrf_scores[doc_id], reverse=True)

    fused = []
    for doc_id in sorted_ids:
        entry = dict(match_data[doc_id])  # shallow copy
        entry["_rrf_score"] = round(rrf_scores[doc_id], 6)
        entry["_retrieval_sources"] = source_info[doc_id]
        fused.append(entry)

    return fused


# ─────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────
def _format_for_llm(products: list[dict]) -> str:
    """Turn structured results into a compact list the LLM can reason over."""
    if not products:
        return "No matching products found."

    lines = []
    for i, p in enumerate(products, 1):
        price = f"${p['price']}" if p.get("price") is not None else "price N/A"
        desc  = (p.get("description") or "")[:160]
        lines.append(
            f"{i}. {p['title']} | {p.get('category','?')} | "
            f"{p.get('brand','?')} | {price}\n   {desc}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Match → product dict conversion
# ─────────────────────────────────────────────
def _match_to_product(match: dict, retrieval_score: float) -> dict:
    """Convert a raw Pinecone match dict into a structured product dict."""
    meta = match.get("metadata", {}) or {}
    asin = match.get("id", "")
    return {
        "title":       meta.get("title", "Untitled product"),
        "description": meta.get("description", ""),
        "category":    meta.get("category", ""),
        "brand":       meta.get("brand", ""),
        "price":       meta.get("price"),
        "currency":    meta.get("currency", "USD"),
        "image_url":   meta.get("image_url", ""),
        "product_url": f"https://www.amazon.com/dp/{asin}" if asin else "",
        "avg_rating":  meta.get("avg_rating"),
        "rating_count": meta.get("rating_count"),
        "retrieval_score": retrieval_score,
    }


# ─────────────────────────────────────────────
# Main search tool
# ─────────────────────────────────────────────
@tool
def search_products(query: str) -> str:
    """Search the product catalogue for items matching the shopper's request.

    Use this whenever the shopper is looking for something to buy, describes a
    need, budget, occasion, or style, or asks for recommendations. The `query`
    should be a natural-language description of what they want
    (e.g. "wireless headphones under $50" or "a comforting gift for a friend").
    """
    session_id = _session_var.get()
    try:
        dense_vec = _embed_query(query)
        sparse_vec = _sparse_encode_query(query)

        # ── Leg 1: Dense-only retrieval (pure semantic ranking) ──
        dense_res = _index.query(
            vector=dense_vec, top_k=CANDIDATE_K, include_metadata=True,
        )
        dense_matches = dense_res.get("matches", [])

        # ── Leg 2: Hybrid retrieval (dense + sparse in one call) ──
        # Pinecone dense-type indexes require a dense vector in every query.
        # Passing sparse_vector alongside the dense vector makes Pinecone
        # combine both signals, producing a keyword-boosted ranking.
        hybrid_matches = []
        if sparse_vec is not None:
            try:
                hybrid_res = _index.query(
                    vector=dense_vec,
                    sparse_vector=sparse_vec,
                    top_k=CANDIDATE_K,
                    include_metadata=True,
                )
                hybrid_matches = hybrid_res.get("matches", [])
            except Exception as e:
                log.warning(f"Hybrid query failed, using dense-only: {e}")

        # ── RRF fusion ──
        if hybrid_matches:
            fused_matches = _rrf_fuse(dense_matches, hybrid_matches)
        else:
            # Fallback: dense-only (no sparse model or hybrid query failed)
            fused_matches = []
            for m in dense_matches:
                d = _to_dict(m)
                d["_rrf_score"] = 0.0
                d["_retrieval_sources"] = {"dense"}
                fused_matches.append(d)

        # Cap at CANDIDATE_K after fusion (dedup may have expanded the list)
        fused_matches = fused_matches[:CANDIDATE_K]

        # ── Build product dicts & apply MIN_SCORE filter ──
        candidates = []
        for match in fused_matches:
            score = round(float(match.get("score", 0.0)), 3)
            if score < MIN_SCORE:
                continue
            product = _match_to_product(match, score)
            candidates.append(product)

        # ── Take TOP_K best results (already ranked by RRF score) ──
        products = candidates[:TOP_K]
        for product in products:
            product["score"] = product.get("retrieval_score", 0.0)

        # Capture structured results so /chat can return product cards
        PRODUCT_BUFFER[session_id] = products

        # ── Logging ──
        n_dense  = len(dense_matches)
        n_hybrid = len(hybrid_matches)
        n_fused  = len(fused_matches)
        n_candidates = len(candidates)
        log.info(
            f"[{session_id}] search '{query}' → "
            f"dense={n_dense}, hybrid={n_hybrid}, fused={n_fused}, "
            f"post-filter={n_candidates}, final→{len(products)}"
        )

        return _format_for_llm(products)

    except Exception as e:
        log.exception(f"[{session_id}] product search failed")
        PRODUCT_BUFFER[session_id] = []
        return "The product search is temporarily unavailable. Apologize briefly and ask the shopper to try again."
