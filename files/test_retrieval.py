"""
test_retrieval.py
=================
Quick sanity-check for the hybrid retrieval pipeline.

Runs a few test queries through the full pipeline:
  Dense top-20 + Sparse top-20 → RRF fuse → top-20 → rerank → top-3

Usage:
    python test_retrieval.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone

load_dotenv()

TOP_K               = 3
CANDIDATE_K         = 20
RRF_K               = 60
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL      = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
BM25_MODEL_PATH     = Path(os.getenv(
    "BM25_MODEL_PATH",
    str(Path(__file__).parent / "bm25_model.json"),
))

TEST_QUERIES = [
    "I need a good laptop for video editing",
    "comfortable running shoes for women",
    "organic green tea bags",
    "yoga mat for home workouts",
    "wireless bluetooth headphones under $50",
    "gift for someone who is feeling sad and needs comfort",   # ← emotion-aware query
]


def load_embedding_model() -> SentenceTransformer:
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    return SentenceTransformer(model_name)


def embed_query(text: str, model: SentenceTransformer) -> list[float]:
    return model.encode([text], normalize_embeddings=True)[0].tolist()


def rrf_fuse(dense_matches, sparse_matches, k=RRF_K):
    """Reciprocal Rank Fusion of two ranked match lists."""
    from collections import defaultdict
    rrf_scores = defaultdict(float)
    match_data = {}
    source_info = defaultdict(set)

    for rank, m in enumerate(dense_matches, 1):
        doc_id = m.get("id", "")
        rrf_scores[doc_id] += 1.0 / (k + rank)
        match_data[doc_id] = m
        source_info[doc_id].add("dense")

    for rank, m in enumerate(sparse_matches, 1):
        doc_id = m.get("id", "")
        rrf_scores[doc_id] += 1.0 / (k + rank)
        if doc_id not in match_data:
            match_data[doc_id] = m
        source_info[doc_id].add("sparse")

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    fused = []
    for doc_id in sorted_ids:
        match = dict(match_data[doc_id])
        match["_rrf_score"] = round(rrf_scores[doc_id], 6)
        match["_sources"] = source_info[doc_id]
        fused.append(match)
    return fused


def main():
    model    = load_embedding_model()
    reranker = CrossEncoder(RERANKER_MODEL)
    pc       = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index    = pc.Index(os.getenv("PINECONE_INDEX_NAME", "shopping-products"))

    # Load BM25 for sparse queries
    bm25 = None
    if BM25_MODEL_PATH.exists():
        from pinecone_text.sparse import BM25Encoder
        bm25 = BM25Encoder().load(str(BM25_MODEL_PATH))
        print(f"✓ BM25 model loaded from {BM25_MODEL_PATH}")
    else:
        print(f"⚠ BM25 model not found at {BM25_MODEL_PATH} — sparse leg disabled")

    stats = index.describe_index_stats()
    print(f"Index has {stats['total_vector_count']} vectors\n")

    for query in TEST_QUERIES:
        print(f"{'═'*70}")
        print(f"Query: {query}")
        print(f"{'─'*70}")

        # Leg 1: Dense
        vector = embed_query(query, model)
        dense_res = index.query(
            vector=vector, top_k=CANDIDATE_K, include_metadata=True,
        )
        dense_matches = dense_res.get("matches", [])

        # Leg 2: Sparse (if BM25 available)
        sparse_matches = []
        if bm25 is not None:
            sparse_vec = bm25.encode_queries(query)
            sparse_res = index.query(
                sparse_vector=sparse_vec, top_k=CANDIDATE_K, include_metadata=True,
            )
            sparse_matches = sparse_res.get("matches", [])

        # Fuse
        if sparse_matches:
            fused = rrf_fuse(dense_matches, sparse_matches)
        else:
            fused = [
                {**m, "_rrf_score": 0.0, "_sources": {"dense"}}
                for m in dense_matches
            ]
        fused = fused[:CANDIDATE_K]

        print(f"  Dense: {len(dense_matches)} | Sparse: {len(sparse_matches)} | Fused: {len(fused)}")

        # Rerank the fused candidates
        pairs = []
        for m in fused:
            meta = m.get("metadata", {})
            parts = [
                meta.get("title", ""),
                meta.get("brand", ""),
                meta.get("category", ""),
                meta.get("description", ""),
            ]
            text = " | ".join(p for p in parts if p).strip()
            pairs.append([query, text])

        if pairs:
            scores = reranker.predict(pairs)
            for m, s in zip(fused, scores):
                m["_rerank_score"] = float(s)
            fused.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)

        # Show top results
        print(f"  Top-{TOP_K} after reranking:")
        for i, match in enumerate(fused[:TOP_K], 1):
            meta    = match.get("metadata", {})
            cosine  = match.get("score", 0)
            rrf     = match.get("_rrf_score", 0)
            rerank  = match.get("_rerank_score", 0)
            sources = match.get("_sources", set())
            price   = f"${meta.get('price', 'N/A')}" if meta.get("price") else "N/A"
            src_tag = "+".join(sorted(sources))
            print(
                f"  {i}. [{meta.get('category','?')}] {meta.get('title','?')[:60]}\n"
                f"     Brand: {meta.get('brand','?')} | Price: {price}\n"
                f"     Cosine: {cosine:.3f} | RRF: {rrf:.4f} | Rerank: {rerank:.4f} | Source: {src_tag}"
            )
        print()


if __name__ == "__main__":
    main()
