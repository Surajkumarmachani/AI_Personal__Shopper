"""
ingest_to_pinecone.py
=====================
Loads product data from Amazon Reviews 2023 (McAuley-Lab) across four domains:
  - Fashion     (Clothing, Shoes & Jewelry)
  - Electronics
  - Grocery     (Grocery & Gourmet Food)
  - Lifestyle   (Health & Household + Sports & Outdoors + Home & Kitchen)

Then normalizes all rows into a unified schema, embeds with a local
SentenceTransformer model, and upserts into a Pinecone serverless index.

REQUIREMENTS
------------
pip install datasets pinecone python-dotenv tqdm sentence-transformers torch

ENV VARS (put in .env file or export manually)
-----------------------------------------------
EMBED_MODEL=BAAI/bge-small-en-v1.5   # or llama-text-embed-v2 / another HF embedding model
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=shopping-products   # will be created if it doesn't exist
PINECONE_CLOUD=aws                      # aws | gcp | azure
PINECONE_REGION=us-east-1              # region matching your cloud

TUNING CONSTANTS (edit these to control cost & speed)
------------------------------------------------------
See the CONFIG block below.
"""

import os
import re
import time
import uuid
import json
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm
import requests as http_requests
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# ─────────────────────────────────────────────
# CONFIG  — edit these to taste
# ─────────────────────────────────────────────
SAMPLES_PER_CATEGORY = 30000       # rows to pull per category slice (~100k+ vectors total after filtering)
DEFAULT_EMBED_MODEL   = "text-embedding-3-small"
DEFAULT_EMBED_DIM     = 1536        # text-embedding-3-small output dimension
UPSERT_BATCH_SIZE     = 100         # Pinecone upsert batch
EMBED_BATCH_SIZE      = 2048        # OpenAI embeddings API max INPUTS per call
EMBED_TOKEN_BUDGET    = 250_000     # OpenAI's real cap is 300k TOKENS per call.
                                    # Batches split on whichever limit hits first.
PER_ITEM_TOKEN_PAD    = 48          # The API counts ~20-25 tokens MORE per input
                                    # than tiktoken (per-input overhead) — that's
                                    # why 280k-budgeted batches still hit 305k+.
                                    # Pad each item's count to stay safely under.

# Categories to load: (display_name, hf_repo_filepath, category_label)
CATEGORIES = [
    # Fashion
    ("Fashion",     "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl",  "Fashion"),
    # Electronics
    ("Electronics", "raw/meta_categories/meta_Electronics.jsonl",                  "Electronics"),
    # Grocery
    ("Grocery",     "raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl",    "Grocery"),
    # Lifestyle — three sub-categories, all labelled "Lifestyle"
    ("Health",      "raw/meta_categories/meta_Health_and_Household.jsonl",        "Lifestyle"),
    ("Sports",      "raw/meta_categories/meta_Sports_and_Outdoors.jsonl",         "Lifestyle"),
    ("Home",        "raw/meta_categories/meta_Home_and_Kitchen.jsonl",            "Lifestyle"),
]

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_text(val) -> str:
    """Flatten lists / dicts to a clean string."""
    if val is None:
        return ""
    if isinstance(val, list):
        val = " ".join(str(v) for v in val if v)
    if isinstance(val, dict):
        val = json.dumps(val)
    text = str(val).strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def parse_price(val) -> Optional[float]:
    """Extract a float from messy price strings like '$12.99' or 'None'."""
    if not val or str(val).lower() in ("none", "null", ""):
        return None
    # strip currency symbols and commas
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_image_url(images_field) -> str:
    """Pull the best available image URL from the images field.
    Handles both formats:
      - dict:  {'hi_res': [...], 'large': [...], 'thumb': [...]}
      - list:  [{'hi_res': '...', 'large': '...', 'thumb': '...', 'variant': '...'}, ...]
    """
    if not images_field:
        return ""
    if isinstance(images_field, str):
        return images_field

    # Raw JSONL format: list of dicts, each with hi_res/large/thumb keys
    if isinstance(images_field, list):
        for item in images_field:
            if isinstance(item, dict):
                for key in ("hi_res", "large", "thumb"):
                    url = item.get(key)
                    if url and str(url).startswith("http"):
                        return str(url)
            elif isinstance(item, str) and item.startswith("http"):
                return item
        return ""

    # HuggingFace schema: {'hi_res': [...], 'large': [...], 'thumb': [...]}
    if isinstance(images_field, dict):
        for key in ("hi_res", "large", "thumb"):
            urls = images_field.get(key, [])
            if isinstance(urls, list):
                for url in urls:
                    if url and str(url).startswith("http"):
                        return str(url)
            elif urls and str(urls).startswith("http"):
                return str(urls)
    return ""


def normalize_row(row: dict, category_label: str) -> Optional[dict]:
    """
    Map a raw McAuley-Lab metadata row → unified product dict.
    Returns None if the row is too thin to be useful (no title/description).
    """
    title       = clean_text(row.get("title"))
    description = clean_text(row.get("description"))
    features    = clean_text(row.get("features"))
    brand       = clean_text(row.get("store") or row.get("brand") or "")
    price       = parse_price(row.get("price"))
    image_url   = extract_image_url(row.get("images"))
    asin        = str(row.get("parent_asin") or row.get("asin") or uuid.uuid4().hex)

    # Build a rich description for embedding
    rich_text = " ".join(filter(None, [title, description, features])).strip()

    # Skip rows with no useful text (can't embed meaningfully)
    if not title and not description:
        return None

    # Skip rows with no price
    if price is None:
        return None

    # Rating info (may be absent)
    avg_rating     = row.get("average_rating")
    rating_number  = row.get("rating_number")

    return {
        # ── Fields stored as Pinecone metadata ──────────────
        "id":           asin,
        "title":        title[:500],        # Pinecone metadata value limit
        "description":  description[:1000],
        "category":     category_label,
        "brand":        brand[:200],
        "price":        price,
        "currency":     "USD",
        "image_url":    image_url[:500],
        "avg_rating":   float(avg_rating) if avg_rating else None,
        "rating_count": int(rating_number) if rating_number else None,
        "source":       "amazon-reviews-2023",
        # ── Text used for embedding (NOT stored in metadata) ─
        "_embed_text":  rich_text[:4000],   # well within token limits
    }


def load_category(display_name: str, hf_filepath: str,
                  category_label: str, n: int) -> list[dict]:
    """
    Stream the raw JSONL file from HuggingFace over HTTP, reading only
    enough lines to collect `n` products. Stops early — no need to
    download the entire multi-GB file.
    """
    log.info(f"Loading {display_name} ({hf_filepath}) — sampling {n} rows …")
    products = []
    skipped  = 0

    # Build the direct download URL for the raw file
    url = (
        f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        f"/resolve/main/{hf_filepath}"
    )

    try:
        # Stream the response — only downloads chunks as we read them
        resp = http_requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()

        seen_ids = set()
        # iter_lines decodes and yields one line at a time
        for raw_line in resp.iter_lines(decode_unicode=True):
            if len(products) >= n:
                break
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            product = normalize_row(row, category_label)
            if product is None:
                skipped += 1
                continue
            # deduplicate by ASIN within this load call
            if product["id"] in seen_ids:
                continue
            seen_ids.add(product["id"])
            products.append(product)

        resp.close()  # stop downloading the rest

    except Exception as e:
        log.error(f"Failed to load {display_name}: {e}")
        return []

    log.info(
        f"  ✓ {len(products)} usable products loaded, {skipped} rows skipped"
    )
    return products


# ─────────────────────────────────────────────
# DENSE EMBEDDING
# ─────────────────────────────────────────────

# cl100k_base is the tokenizer used by text-embedding-3-small/large and GPT-4o.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _make_batches(texts: list[str], max_items: int, max_tokens: int) -> list[list[int]]:
    """
    Split text indices into batches that respect BOTH limits:
      - at most `max_items` texts per batch (API cap: 2048 inputs/request)
      - at most `max_tokens` total tokens per batch (API cap: 300k tokens/request)

    A single text longer than max_tokens gets its own batch (best effort —
    the API call for it may still fail, but nothing else is blocked by it).
    Returns lists of indices into `texts`, not the texts themselves.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for i, text in enumerate(texts):
        # Pad per item: the API's server-side count runs ~20-25 tokens/input
        # higher than tiktoken's, so budgeting on raw counts overshoots.
        t = _count_tokens(text) + PER_ITEM_TOKEN_PAD
        would_exceed_items  = len(current) + 1 > max_items
        would_exceed_tokens = current_tokens + t > max_tokens
        if current and (would_exceed_items or would_exceed_tokens):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += t

    if current:
        batches.append(current)
    return batches


def embed_products(products: list[dict], model_name: str, client: OpenAI) -> list[dict]:
    """
    Add an 'embedding' key to each product dict using the OpenAI Embeddings API.

    Batches on BOTH the item-count limit (2048/request) and the token-budget
    limit (300k tokens/request) — a batch of 2048 long product descriptions can
    easily blow past the token cap even though it's under the item cap, which
    is exactly what caused the earlier `max_tokens_per_request` 400 errors.
    """
    texts = [p["_embed_text"] for p in products]
    all_embeddings: list[list[float] | None] = [None] * len(texts)

    batches = _make_batches(texts, EMBED_BATCH_SIZE, EMBED_TOKEN_BUDGET)
    log.info(
        f"Embedding {len(texts)} products with {model_name} in {len(batches)} "
        f"batches (≤{EMBED_BATCH_SIZE} items / ≤{EMBED_TOKEN_BUDGET} tokens each) …"
    )

    for batch_indices in tqdm(batches, desc="Embedding"):
        batch = [texts[i] for i in batch_indices]
        try:
            response = client.embeddings.create(
                model=model_name,
                input=batch,
            )
            # Response items are returned in the same order as input
            for idx, item in zip(batch_indices, response.data):
                all_embeddings[idx] = item.embedding
        except Exception as e:
            log.error(f"Embedding error on batch starting at index {batch_indices[0]}: {e}")
            # fill with zeros so we don't lose the products; they'll upsert but
            # won't retrieve well — re-run to fix
            for idx in batch_indices:
                all_embeddings[idx] = [0.0] * DEFAULT_EMBED_DIM

    for product, embedding in zip(products, all_embeddings):
        product["embedding"] = embedding if embedding is not None else [0.0] * DEFAULT_EMBED_DIM

    return products


# ─────────────────────────────────────────────
# SPARSE EMBEDDING (BM25)
# ─────────────────────────────────────────────

def fit_bm25(products: list[dict], save_path: str) -> BM25Encoder:
    """
    Fit a BM25 encoder on the corpus and save the fitted model to disk.
    The search module loads this same model at startup so its query-time
    sparse vectors are consistent with the corpus vocabulary/IDF weights.
    """
    texts = [p["_embed_text"] for p in products]
    log.info(f"Fitting BM25 encoder on {len(texts)} documents …")
    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump(save_path)
    log.info(f"  ✓ BM25 model saved to {save_path}")
    return bm25


def sparse_encode_products(products: list[dict], bm25: BM25Encoder) -> list[dict]:
    """
    Add a 'sparse_values' key to each product dict using the fitted BM25 encoder.
    The sparse vector is the {indices, values} dict Pinecone expects.
    """
    texts = [p["_embed_text"] for p in products]
    log.info(f"Generating sparse vectors for {len(texts)} products …")
    sparse_vectors = bm25.encode_documents(texts)
    for product, sparse_vec in zip(products, sparse_vectors):
        product["sparse_values"] = sparse_vec
    log.info("  ✓ Sparse vectors generated.")
    return products


# ─────────────────────────────────────────────
# PINECONE UPSERT
# ─────────────────────────────────────────────

def build_pinecone_record(product: dict) -> dict:
    """Convert a product dict into a Pinecone upsert record.

    Includes both dense (`values`) and sparse (`sparse_values`) vectors
    for hybrid retrieval.
    """
    # Only include non-None metadata fields
    metadata = {
        k: v for k, v in {
            "title":        product["title"],
            "description":  product["description"],
            "category":     product["category"],
            "brand":        product["brand"],
            "price":        product["price"],
            "currency":     product["currency"],
            "image_url":    product["image_url"],
            "avg_rating":   product["avg_rating"],
            "rating_count": product["rating_count"],
            "source":       product["source"],
        }.items()
        if v is not None and v != ""
    }

    record = {
        "id":       product["id"],
        "values":   product["embedding"],
        "metadata": metadata,
    }

    # Attach sparse vector if available (generated by BM25 encoder)
    if "sparse_values" in product and product["sparse_values"]:
        record["sparse_values"] = product["sparse_values"]

    return record


def upsert_to_pinecone(products: list[dict], index) -> None:
    """Upsert in batches, showing a progress bar."""
    records = [build_pinecone_record(p) for p in products]

    log.info(f"Upserting {len(records)} records to Pinecone …")
    for i in tqdm(range(0, len(records), UPSERT_BATCH_SIZE), desc="Upserting"):
        batch = records[i : i + UPSERT_BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            log.error(f"Upsert error on batch {i}: {e}")


# ─────────────────────────────────────────────
# PINECONE INDEX SETUP
# ─────────────────────────────────────────────

def get_or_create_index(pc: Pinecone, index_name: str,
                        cloud: str, region: str, embed_dim: int):
    """Create the index if it doesn't already exist, or recreate if config mismatches.

    Uses dotproduct metric (required for sparse/hybrid vector support).
    For normalized embeddings (bge-small uses normalize_embeddings=True),
    dotproduct and cosine produce identical rankings.
    """
    target_metric = "dotproduct"
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        desc = pc.describe_index(index_name)
        needs_recreate = False
        reason = ""

        if desc.dimension != embed_dim:
            needs_recreate = True
            reason = (
                f"dimension mismatch: index has {desc.dimension}, "
                f"model needs {embed_dim}"
            )
        elif desc.metric != target_metric:
            needs_recreate = True
            reason = (
                f"metric mismatch: index uses '{desc.metric}', "
                f"hybrid retrieval needs '{target_metric}'"
            )

        if needs_recreate:
            log.info(f"Index '{index_name}': {reason}. Deleting and recreating …")
            pc.configure_index(index_name, deletion_protection="disabled")
            pc.delete_index(index_name)
            time.sleep(5)
        else:
            log.info(f"Index '{index_name}' already exists — reusing.")
            return pc.Index(index_name)

    log.info(f"Creating Pinecone index '{index_name}' (dim={embed_dim}, metric={target_metric}) …")
    pc.create_index(
        name=index_name,
        dimension=embed_dim,
        metric=target_metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    # wait for it to become ready
    while not pc.describe_index(index_name).status["ready"]:
        log.info("  Waiting for index to be ready …")
        time.sleep(2)
    log.info("  ✓ Index ready.")


    return pc.Index(index_name)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()

    pinecone_key   = os.getenv("PINECONE_API_KEY")
    openai_key     = os.getenv("OPENAI_API_KEY")
    index_name     = os.getenv("PINECONE_INDEX_NAME", "shopping-products")
    cloud          = os.getenv("PINECONE_CLOUD",      "aws")
    region         = os.getenv("PINECONE_REGION",     "us-east-1")
    embed_model_id = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    if not pinecone_key:
        raise EnvironmentError("PINECONE_API_KEY not set")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    embed_dim  = DEFAULT_EMBED_DIM
    openai_client = OpenAI(api_key=openai_key)
    pc         = Pinecone(api_key=pinecone_key)
    index      = get_or_create_index(pc, index_name, cloud, region, embed_dim)

    all_products: list[dict] = []

    # ── Step 1: Load & normalize ──────────────
    seen_global_ids = set()
    for display_name, hf_filepath, category_label in CATEGORIES:
        products = load_category(
            display_name, hf_filepath, category_label, SAMPLES_PER_CATEGORY
        )
        # global dedup across all categories (same ASIN can appear in multiple)
        unique = []
        for p in products:
            if p["id"] not in seen_global_ids:
                seen_global_ids.add(p["id"])
                unique.append(p)
        all_products.extend(unique)

    log.info(
        f"\n{'─'*50}\n"
        f"Total unique products across all domains: {len(all_products)}\n"
        f"{'─'*50}"
    )

    # ── Step 2: Breakdown by category ─────────
    from collections import Counter
    counts = Counter(p["category"] for p in all_products)
    for cat, count in sorted(counts.items()):
        log.info(f"  {cat:<15} {count} products")

    # ── Step 3: Embed (dense) via OpenAI API ──
    log.info(f"Using OpenAI embedding model: {embed_model_id}")
    all_products = embed_products(all_products, embed_model_id, openai_client)

    # ── Step 4: Sparse vectors (BM25) ─────────
    bm25_save_path = os.getenv(
        "BM25_MODEL_PATH",
        str(Path(__file__).parent / "bm25_model.json"),
    )
    bm25 = fit_bm25(all_products, bm25_save_path)
    all_products = sparse_encode_products(all_products, bm25)

    # ── Step 5: Upsert ────────────────────────
    upsert_to_pinecone(all_products, index)

    # ── Step 6: Verify ────────────────────────
    time.sleep(2)  # allow index to reflect new vectors
    stats = index.describe_index_stats()
    log.info(
        f"\n✅ Done! Pinecone index '{index_name}' now has "
        f"{stats['total_vector_count']} vectors."
    )


if __name__ == "__main__":
    main()
