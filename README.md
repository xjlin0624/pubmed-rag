# PubMed RAG — Medical Literature Question Answering

A retrieval-augmented generation (RAG) system for answering medical questions grounded in PubMed abstracts. Built for CS 6120 Natural Language Processing, Northeastern University.

---

## Overview

This system retrieves relevant PubMed abstracts for a user query and generates a cited, faithful answer using a locally-served LLM. Every claim in the generated answer is verified against retrieved passages using an NLI-based faithfulness checker.

```
User query → Hybrid Retriever (BM25 + Dense) → Local LLM (Ollama) → Faithfulness Check → Cited Answer
```

---

## System architecture

Diagram source (Mermaid): [`diagrams/system_architecture.mmd`](diagrams/system_architecture.mmd). Preview in the IDE or export to PNG with the [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli).

---

## Requirements

- Docker (recommended for a one-command demo)
- A GCP VM with **≥16 GB RAM** (GPU optional; CPU is fine for smaller models)
- Or a local machine with **≥8 GB RAM** plus Ollama for development

---

## Quickstart (Docker)

### 1. Clone and configure

```bash
git clone https://github.com/your-org/pubmed-rag.git
cd pubmed-rag
cp .env.example .env
```

### 2. Demo without PubMed indexes (mock retriever)

```bash
docker build -t pubmed-rag .
docker run --rm --env-file .env -e USE_MOCK_RETRIEVER=1 -p 8501:8501 pubmed-rag
```

First boot downloads Ollama weights and the Hugging Face NLI model (one-time, several GB). Open `http://localhost:8501`.

### 3. Full stack with hybrid retrieval

Build indexes on the host, then mount them:

```bash
pip install -r requirements.txt
python -m retriever.retriever
docker build -t pubmed-rag .
docker run --rm --env-file .env -p 8501:8501 \
  -v "$(pwd)/chunks.json:/app/chunks.json:ro" \
  -v "$(pwd)/faiss.index:/app/faiss.index:ro" \
  pubmed-rag
```

Or set `BUILD_RETRIEVAL_INDEX=1` in `.env` so the container runs `python -m retriever.retriever` on startup (slow; needs stable outbound network).

---

## Running without Docker

```bash
pip install -r requirements.txt
ollama serve
ollama pull qwen3:4b
python -m retriever.retriever
streamlit run app.py --server.port 8501
```

CLI smoke test: `python pipeline.py`.

---

## GCP (demo day)

1. Use an Ubuntu (or similar) VM with **≥16 GB RAM**; attach a **static external IP** if the demo URL must stay fixed.
2. In VPC networking / firewalls, allow **ingress TCP 8501** to the instance (scoped to your class IPs, or `0.0.0.0/0` for an open class demo).
3. Install Docker, clone the repo, `docker build` / `docker run` with `-p 8501:8501`. Ollama listens on `127.0.0.1` inside the VM; only Streamlit needs to be reachable on the public IP.
4. Optional GPU: install the NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then `docker run --gpus all ...`.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b` | Model used via Ollama |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `FAITHFULNESS_THRESHOLD` | `0.5` | NLI entailment cutoff per sentence |
| `MIN_RETRIEVAL_SCORE` | `0.3` | Minimum top retrieval score before generator fallback |
| `MAX_TOKENS` | `300` | `num_predict` cap for Ollama |
| `TOP_K` | `5` | Passages fed to the generator |
| `BM25_ALPHA` | `0.5` | Hybrid fusion (`1` = BM25 only, `0` = dense only) |
| `CHUNKS_FILE` / `INDEX_FILE` | `chunks.json` / `faiss.index` | Paths checked before calling the real retriever |
| `USE_MOCK_RETRIEVER` | `0` | Set to `1` to force mock passages |
| `BUILD_RETRIEVAL_INDEX` | `0` | Set to `1` in Docker to auto-build indexes at startup |

---

## Project structure

```
pubmed-rag/
├── app.py
├── pipeline.py
├── Dockerfile
├── entrypoint.sh
├── requirements.txt
├── .env.example
├── diagrams/
│   └── system_architecture.mmd
├── retriever/
│   └── retriever.py          # ingest + retrieve(query, top_k)
├── generator/
│   └── generator.py          # generate_answer, check_faithfulness
└── evaluation/
    └── ablation_smoke.py     # four-branch smoke table + JSON
```

---

## Data

`python -m retriever.retriever` downloads PubMed XML via the [NLM E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25497/), parses abstracts, chunks by sentence, and builds a FAISS index. The search query and `MAX_RESULTS` are configured at the top of `retriever/retriever.py`. Step 1 is rate-limited; a 10k crawl can take tens of minutes.

Chunks carry PMID, sentence text, position, year, and MeSH terms. Full BioASQ benchmarks can be plugged in by the evaluation teammate; this repo ships a **smoke ablation** runner for integration.

---

## Ablation coordination

Four integration branches (BM25-only, dense-only, hybrid, hybrid + stricter NLI) share one command. JSON is written to `evaluation/ablation_smoke_results.json` and a Markdown table is printed.

```bash
python evaluation/ablation_smoke.py
```

When the mock retriever is active, changing `BM25_ALPHA` does not change passages; use real indexes to measure retrieval differences. Replace the placeholders below with team metrics (e.g. BioASQ EM / F1 / Recall@k) when available.

| Strategy | Recall@5 | F1 | Faithfulness |
|---|---|---|---|
| BM25-only (`BM25_ALPHA=1`) | TBD | TBD | TBD |
| Dense-only (`BM25_ALPHA=0`) | TBD | TBD | TBD |
| Hybrid (`BM25_ALPHA=0.5`) | TBD | TBD | TBD |
| Hybrid + strict NLI (`ABLATION_STRICT_NLI`) | TBD | TBD | TBD |

---

## Team

| Member | Role |
|---|---|
| Xinyi Jiang | Integration, infrastructure, Docker, GCP deployment |
| Minjia Fang | Data pipeline, retrieval (BM25, dense, hybrid) |
| Xuelan Lin | Local LLM deployment, generation, faithfulness verification |
| Weiyi Sun | Streamlit frontend, evaluation, EDA |

---

## Acknowledgements

- [PubMed / NLM E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/) — abstract corpus
- [BioASQ](http://bioasq.org/) — evaluation dataset
- [Ollama](https://ollama.com/) — local LLM serving
- [Qwen3](https://huggingface.co/Qwen) — base language model
- [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) — faithfulness NLI model
