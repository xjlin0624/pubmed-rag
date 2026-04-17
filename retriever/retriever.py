"""
Module B — Data & Retrieval Pipeline
======================================
Steps 1-4 build the indexes (run once before serving queries):
  1  Download raw abstracts from PubMed API  → raw_abstracts.json
  2  Parse XML, extract structured fields    → abstracts.json
  3  Chunk abstracts into sentences          → chunks.json
  4  Build FAISS vector index                → embeddings.npy, faiss.index

Step 5 is the runtime retrieve() interface used by pipeline.py.

Usage:
    python -m retriever.retriever                  # Build all indexes
    python -m retriever.retriever --step 1         # Single step
"""

import argparse
import json
import os
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
EMBED_MODEL     = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Default fusion weight; actual value is read per retrieve() so ablations can vary BM25_ALPHA without reloading.
BM25_ALPHA_DEFAULT = float(os.getenv("BM25_ALPHA", "0.5"))

RAW_FILE        = os.getenv("RAW_FILE", "raw_abstracts.json")
ABSTRACTS_FILE  = os.getenv("ABSTRACTS_FILE", "abstracts.json")
CHUNKS_FILE     = os.getenv("CHUNKS_FILE", "chunks.json")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "embeddings.npy")
INDEX_FILE      = os.getenv("INDEX_FILE", "faiss.index")


# ── Step 1: Download PubMed Data ───────────────────────────────────────────────

def search_pmids(query, max_results):
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    r = requests.get(PUBMED_BASE_URL + "esearch.fcgi", params=params)
    return r.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids, batch_size):
    results = []
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching batches"):
        batch = pmids[i:i + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "rettype": "abstract"}
        r = requests.get(PUBMED_BASE_URL + "efetch.fcgi", params=params)
        results.append(r.text)
        time.sleep(0.4)
    return results


def step1_download():
    print("\n── Step 1: Downloading PubMed Data ──────────────────────────────")
    pmids = search_pmids(SEARCH_QUERY, MAX_RESULTS)
    print(f"Found {len(pmids)} records")
    raw = fetch_abstracts(pmids, BATCH_SIZE_API)
    with open(RAW_FILE, "w") as f:
        json.dump(raw, f)
    print(f"Saved {len(raw)} batches → {RAW_FILE}")


# ── Step 2: Parse XML ──────────────────────────────────────────────────────────

def parse_xml_batch(xml_str):
    records = []
    root = ET.fromstring(xml_str)
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID")
            year = article.findtext(".//PubDate/Year") or \
                   article.findtext(".//PubDate/MedlineDate", "")[:4]
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join([a.text or "" for a in abstract_parts]).strip()
            mesh_terms = [
                m.text for m in article.findall(".//MeshHeading/DescriptorName") if m.text
            ]
            if not abstract:
                continue
            records.append({"pmid": pmid, "year": year, "abstract": abstract, "mesh": mesh_terms})
        except Exception:
            continue
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
        if len(sent) < 20:
            continue
        chunks.append({
            "pmid":     record["pmid"],
            "year":     record["year"],
            "mesh":     record["mesh"],
            "chunk_id": f"{record['pmid']}_{i}",
            "position": i,
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
    model = SentenceTransformer(EMBED_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE_ENC, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors, dimension {dim}")
    np.save(EMBEDDINGS_FILE, embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved → {EMBEDDINGS_FILE}, {INDEX_FILE}")


# ── Step 5: Runtime Retrieval Interface ───────────────────────────────────────
# Module-level state — loaded once on first retrieve() call
_bm25   = None
_chunks = None
_index  = None
_model  = None


def _load():
    global _bm25, _chunks, _index, _model
    with open(CHUNKS_FILE, "r") as f:
        _chunks = json.load(f)
    tokenized = [c["text"].lower().split() for c in _chunks]
    _bm25 = BM25Okapi(tokenized)
    _index = faiss.read_index(INDEX_FILE)
    _model = SentenceTransformer(EMBED_MODEL)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid BM25 + dense retrieval.

    Args:
        query:  Natural language search query.
        top_k:  Number of results to return.

    Returns:
        List of dicts: {"pmid": str, "text": str, "score": float, ...}
    """
    global _bm25, _chunks, _index, _model
    if _bm25 is None:
        _load()

    n = len(_chunks)

    bm25_scores = _bm25.get_scores(query.lower().split())
    bm25_max = bm25_scores.max() or 1.0
    bm25_scores = bm25_scores / bm25_max

    dense_scores = np.zeros(n)
    query_vec = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = _index.search(query_vec, top_k * 10)
    for score, idx in zip(scores[0], indices[0]):
        dense_scores[idx] = float(score)

    alpha = float(os.getenv("BM25_ALPHA", str(BM25_ALPHA_DEFAULT)))
    alpha = min(1.0, max(0.0, alpha))
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores
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
            "text":     _chunks[idx]["text"],
        }
        for idx in top_indices
    ]


# ── Step 5: Test Retrieval ─────────────────────────────────────────────────────

def step5_test(query):
    print("\n── Step 5: Testing Retrieval Interface ───────────────────────────")
    print(f"Query: {query}")
    print("-" * 60)
    results = retrieve(query, top_k=5)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r['score']}")
        print(f"     PMID: {r['pmid']}")
        print(f"     {r['text']}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module B — Build retrieval indexes")
    parser.add_argument("--step", type=int, default=0,
                        help="Run a single step (1–5). Default: run all steps.")
    parser.add_argument("--query", type=str,
                        default="insulin resistance in diabetic patients",
                        help="Query string for Step 5 test.")
    args = parser.parse_args()

    steps = {1: step1_download, 2: step2_parse, 3: step3_chunk, 4: step4_build_index}

    if args.step == 0:
        for fn in steps.values():
            fn()
        step5_test(args.query)
    elif args.step in steps:
        steps[args.step]()
    elif args.step == 5:
        step5_test(args.query)
    else:
        print("Invalid step. Choose from 1–5.")
