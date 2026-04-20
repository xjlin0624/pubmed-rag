# PubMed RAG — Medical Literature Question Answering

A retrieval-augmented generation (RAG) system for answering medical questions grounded in PubMed abstracts. Built for CS 6120 Natural Language Processing, Northeastern University.

---

## Overview

This system retrieves relevant PubMed abstracts for a user query and generates a cited, faithful answer using a locally-served LLM. Every claim in the generated answer is verified against retrieved passages using an NLI-based faithfulness checker. If the answer is not sufficiently supported, the pipeline iteratively re-retrieves additional context using unsupported sentences as follow-up queries.

```
User query → Hybrid Retriever (BM25 + MedCPT Dense) → Local LLM (Ollama/qwen2.5:7b)
           → Faithfulness Check (NLI) → [Re-retrieve if needed] → Cited Answer
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
git clone https://github.com/xjlin0624/pubmed-rag.git
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
ollama pull qwen2.5:7b
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
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model used via Ollama |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama chat endpoint |
| `FAITHFULNESS_THRESHOLD` | `0.3` | NLI entailment cutoff per sentence |
| `MIN_RETRIEVAL_SCORE` | `0.3` | Minimum top retrieval score before generator fallback |
| `MAX_TOKENS` | `300` | `num_predict` cap for Ollama |
| `TOP_K` | `5` | Passages fed to the generator per retrieval round |
| `MAX_ITER` | `2` | Max iterative re-retrieval rounds when faithfulness is low |
| `MAX_CONTEXT` | `15` | Max total passages across all retrieval rounds |
| `BM25_ALPHA` | `0.5` | Hybrid fusion (`1` = BM25 only, `0` = dense only) |
| `EMBED_MODEL` | `ncbi/MedCPT-Article-Encoder` | Sentence embedding model for dense retrieval |
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
├── data/
│   └── bioasq_gold.json      # BioASQ gold evaluation questions
├── retriever/
│   └── retriever.py          # ingest + retrieve(query, top_k)
├── generator/
│   └── generator.py          # generate_answer, check_faithfulness
└── evaluation/
    ├── run_eval.py           # run pipeline over BioASQ, save predictions
    ├── ablation_smoke.py     # four-branch smoke table + JSON
    ├── bioasq_eval.py        # BioASQ EM / F1 / Recall@k metrics
    ├── eda.py                # corpus EDA (matplotlib)
    └── eda_streamlit.py      # interactive EDA dashboard
```

---

## Data

`python -m retriever.retriever` downloads PubMed XML via the [NLM E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25497/), parses abstracts, chunks by sentence, and builds a FAISS index using [MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) embeddings. The search query and `MAX_RESULTS` are configured at the top of `retriever/retriever.py`. Step 1 is rate-limited; a 10k crawl can take tens of minutes.

**Runtime files needed** (build once, copy to demo machine):
- `chunks.json` (~35 MB) — sentence chunks with metadata
- `faiss.index` (~137 MB) — MedCPT dense index

Chunks carry PMID, sentence text, position, year, and MeSH terms. Full BioASQ benchmarks can be plugged in via `evaluation/bioasq_eval.py`.

---

## Ablation results

Four configurations tested on the diabetes corpus:

| Strategy | Faithfulness |
|---|---|
| BM25-only (`BM25_ALPHA=1`) | 0.0 |
| Dense-only (`BM25_ALPHA=0`) | 1.0 |
| Hybrid (`BM25_ALPHA=0.5`) | 1.0 |
| Hybrid + strict NLI (`ABLATION_STRICT_NLI=0.75`) | 0.0 |

Full results in `evaluation/ablation_smoke_results.json`. BioASQ EM / F1 / Recall@k metrics: pending GCP evaluation run.

```bash
python -m evaluation.ablation_smoke
```

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
- [Qwen2.5](https://huggingface.co/Qwen) — base language model
- [ncbi/MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) — domain-specific biomedical embeddings
- [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) — faithfulness NLI model
