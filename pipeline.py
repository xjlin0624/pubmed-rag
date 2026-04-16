"""
Module B — Data & Retrieval Pipeline
=====================================
Runs all 5 steps in sequence:
  Step 1: Download raw abstracts from PubMed API
  Step 2: Parse XML, extract structured fields
  Step 3: Chunk abstracts into sentence-level units
  Step 4: Build FAISS vector index
  Step 5: Test unified hybrid retrieval interface

Usage:
    python pipeline.py                  # Run full pipeline
    python pipeline.py --step 1         # Run a single step
    python pipeline.py --step 5 --query "your query here"
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SEARCH_QUERY    = "diabetes"
MAX_RESULTS     = 10000
BATCH_SIZE_API  = 200
BATCH_SIZE_ENC  = 256
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
BM25_ALPHA      = 0.5   # Hybrid fusion weight: BM25 * alpha + Dense * (1 - alpha)

# File paths
RAW_FILE        = "raw_abstracts.json"
ABSTRACTS_FILE  = "abstracts.json"
CHUNKS_FILE     = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE      = "faiss.index"


# ── Step 1: Download PubMed Data ───────────────────────────────────────────────

def search_pmids(query, max_results):
    """Search PubMed by keyword and return a list of PMIDs"""
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json"
    }
    r = requests.get(PUBMED_BASE_URL + "esearch.fcgi", params=params)
    return r.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids, batch_size):
    """Fetch abstracts in batches given a list of PMIDs"""
    results = []
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching batches"):
        batch = pmids[i:i + batch_size]
        params = {
            "db":      "pubmed",
            "id":      ",".join(batch),  # Join PMIDs with comma
            "retmode": "xml",
            "rettype": "abstract"        # Fetch abstracts only, not full text
        }
        r = requests.get(PUBMED_BASE_URL + "efetch.fcgi", params=params)
        results.append(r.text)
        time.sleep(0.4)                  # Pause between batches to avoid rate limiting
    return results


def step1_download():
    print("\n── Step 1: Downloading PubMed Data ──────────────────────────────")
    print(f"Searching PMIDs for query: '{SEARCH_QUERY}'...")
    pmids = search_pmids(SEARCH_QUERY, MAX_RESULTS)
    print(f"Found {len(pmids)} records")

    print("Fetching abstracts...")
    raw = fetch_abstracts(pmids, BATCH_SIZE_API)

    with open(RAW_FILE, "w") as f:
        json.dump(raw, f)
    print(f"Saved {len(raw)} batches → {RAW_FILE}")


# ── Step 2: Parse XML ──────────────────────────────────────────────────────────

def parse_xml_batch(xml_str):
    """Parse a batch of XML and extract structured fields"""
    records = []
    root = ET.fromstring(xml_str)

    for article in root.findall(".//PubmedArticle"):
        try:
            # Extract PMID
            pmid = article.findtext(".//PMID")

            # Extract publication year
            year = article.findtext(".//PubDate/Year") or \
                   article.findtext(".//PubDate/MedlineDate", "")[:4]

            # Extract abstract text
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join([a.text or "" for a in abstract_parts]).strip()

            # Extract MeSH terms
            mesh_terms = [
                m.text for m in article.findall(".//MeshHeading/DescriptorName")
                if m.text
            ]

            # Skip articles with no abstract
            if not abstract:
                continue

            records.append({
                "pmid":     pmid,
                "year":     year,
                "abstract": abstract,
                "mesh":     mesh_terms
            })
        except Exception:
            continue  # Skip malformed records

    return records


def step2_parse():
    print("\n── Step 2: Parsing XML ───────────────────────────────────────────")
    with open(RAW_FILE, "r") as f:
        raw_batches = json.load(f)

    all_records = []
    for batch in tqdm(raw_batches, desc="Parsing batches"):
        all_records.extend(parse_xml_batch(batch))

    print(f"Parsed {len(all_records)} records with abstracts")

    with open(ABSTRACTS_FILE, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved → {ABSTRACTS_FILE}")


# ── Step 3: Chunk Abstracts ────────────────────────────────────────────────────

def chunk_by_sentence(record):
    """Split abstract into sentence-level chunks, preserving source metadata"""
    sentences = (
        record["abstract"]
        .replace("? ", "?|")
        .replace("! ", "!|")
        .replace(". ", ".|")
        .split("|")
    )

    chunks = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < 20:              # Skip very short fragments
            continue
        chunks.append({
            "pmid":     record["pmid"],
            "year":     record["year"],
            "mesh":     record["mesh"],
            "chunk_id": f"{record['pmid']}_{i}",  # Unique chunk identifier
            "position": i,                          # Position within abstract
            "text":     sent
        })
    return chunks


def step3_chunk():
    print("\n── Step 3: Chunking Abstracts ────────────────────────────────────")
    with open(ABSTRACTS_FILE, "r") as f:
        records = json.load(f)

    all_chunks = []
    for record in tqdm(records, desc="Chunking"):
        all_chunks.extend(chunk_by_sentence(record))

    print(f"Total chunks: {len(all_chunks)}")

    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Saved → {CHUNKS_FILE}")


# ── Step 4: Build Vector Index ─────────────────────────────────────────────────

def step4_build_index():
    print("\n── Step 4: Building Vector Index ────────────────────────────────")
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE_ENC,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS flat inner product index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors, dimension {dim}")

    np.save(EMBEDDINGS_FILE, embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved → {EMBEDDINGS_FILE}, {INDEX_FILE}")


# ── Step 5: Unified Retrieval Interface ───────────────────────────────────────

# Module-level state (loaded once at import / first retrieval call)
_bm25   = None
_chunks = None
_index  = None
_model  = None


def _load_retriever():
    """Load all indexes and models into memory (called once)"""
    global _bm25, _chunks, _index, _model

    print("Loading chunks...")
    with open(CHUNKS_FILE, "r") as f:
        _chunks = json.load(f)

    print("Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in tqdm(_chunks, desc="Tokenizing")]
    _bm25 = BM25Okapi(tokenized)

    print("Loading FAISS index...")
    _index = faiss.read_index(INDEX_FILE)

    print("Loading embedding model...")
    _model = SentenceTransformer(EMBED_MODEL)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Unified retrieval interface.
    Combines BM25 sparse + Dense vector search with linear score fusion.

    Args:
        query:  Natural language search query
        top_k:  Number of results to return

    Returns:
        List of result dicts with fields:
        score, bm25, dense, pmid, year, mesh, chunk_id, position, text
    """
    global _bm25, _chunks, _index, _model
    if _bm25 is None:
        _load_retriever()

    n = len(_chunks)

    # BM25 scores — normalized to [0, 1]
    bm25_scores = _bm25.get_scores(query.lower().split())
    bm25_max = bm25_scores.max() or 1.0
    bm25_scores = bm25_scores / bm25_max

    # Dense scores via FAISS
    dense_scores = np.zeros(n)
    query_vec = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = _index.search(query_vec, top_k * 10)
    for score, idx in zip(scores[0], indices[0]):
        dense_scores[idx] = float(score)

    # Hybrid fusion
    hybrid_scores = BM25_ALPHA * bm25_scores + (1 - BM25_ALPHA) * dense_scores

    # Top-k results
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    return [
        {
            "score":    round(float(hybrid_scores[idx]), 4),
            "bm25":     round(float(bm25_scores[idx]), 4),
            "dense":    round(float(dense_scores[idx]), 4),
            "pmid":     _chunks[idx]["pmid"],
            "year":     _chunks[idx]["year"],
            "mesh":     _chunks[idx]["mesh"],
            "chunk_id": _chunks[idx]["chunk_id"],
            "position": _chunks[idx]["position"],
            "text":     _chunks[idx]["text"]
        }
        for idx in top_indices
    ]


def step5_test(query):
    print("\n── Step 5: Testing Retrieval Interface ───────────────────────────")
    print(f"Query: {query}")
    print("-" * 60)
    results = retrieve(query, top_k=5)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r['score']} (BM25: {r['bm25']} | Dense: {r['dense']})")
        print(f"     PMID: {r['pmid']} | Year: {r['year']}")
        print(f"     {r['text']}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module B — RAG Retrieval Pipeline")
    parser.add_argument("--step",  type=int, default=0,
                        help="Run a single step (1–5). Default: run all steps.")
    parser.add_argument("--query", type=str,
                        default="insulin resistance in diabetic patients",
                        help="Query string for Step 5 test.")
    args = parser.parse_args()

    if args.step == 0:
        step1_download()
        step2_parse()
        step3_chunk()
        step4_build_index()
        step5_test(args.query)
    elif args.step == 1:
        step1_download()
    elif args.step == 2:
        step2_parse()
    elif args.step == 3:
        step3_chunk()
    elif args.step == 4:
        step4_build_index()
    elif args.step == 5:
        step5_test(args.query)
    else:
        print("Invalid step. Choose from 1–5.")
