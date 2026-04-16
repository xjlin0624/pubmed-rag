# PubMed RAG — Medical Literature Question Answering

A retrieval-augmented generation (RAG) system for answering medical questions grounded in PubMed abstracts. Built for CS 6120 Natural Language Processing, Northeastern University.

---

## Overview

This system retrieves relevant PubMed abstracts for a user query and generates a cited, faithful answer using a locally-served LLM. Every claim in the generated answer is verified against retrieved passages using an NLI-based faithfulness checker.

```
User query → Hybrid Retriever (BM25 + Dense) → Local LLM (Ollama) → Faithfulness Check → Cited Answer
```

---

## Requirements

- Docker
- A GCP instance with at least 16GB RAM (GPU recommended)
- Or: local machine with 8GB+ RAM for development

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-org/pubmed-rag.git
cd pubmed-rag
```

### 2. Set up environment variables

```bash
cp .env.example .env
# edit .env if needed — defaults work out of the box
```

### 3. Build and run with Docker

```bash
docker build -t pubmed-rag .
docker run --env-file .env -p 8501:8501 pubmed-rag
```

On first run, the container will:
1. Pull the LLM model via Ollama (~5GB, one-time download)
2. Download and index PubMed abstracts (~10k records)
3. Start the Streamlit frontend on port 8501

Open your browser at `http://localhost:8501`

---

## Running Without Docker (Development)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
ollama serve
```

### 3. Download and index data

```bash
python data/download_pubmed.py      # downloads 10k+ abstracts from PubMed
python data/preprocess.py           # chunks, embeds, and builds FAISS index
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b` | Local model served by Ollama |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `FAITHFULNESS_THRESHOLD` | `0.5` | NLI entailment score cutoff |
| `MIN_RETRIEVAL_SCORE` | `0.3` | Minimum retrieval score before fallback |
| `MAX_TOKENS` | `300` | Max tokens in generated answer |
| `TOP_K` | `5` | Number of passages retrieved per query |

---

## Project Structure

```
pubmed-rag/
├── app.py                  # Streamlit frontend
├── pipeline.py             # end-to-end query pipeline
├── requirements.txt
├── Dockerfile
├── entrypoint.sh
├── .env.example
│
├── data/
│   ├── download_pubmed.py  # fetches abstracts via NLM E-utilities API
│   └── preprocess.py       # chunking, embedding, FAISS index creation
│
├── retriever/
│   ├── bm25.py             # sparse BM25 baseline
│   ├── dense.py            # BioBERT dense retrieval
│   ├── hybrid.py           # BM25 + dense score fusion
│   └── retriever.py        # unified retrieve(query, top_k) interface
│
├── generator/
│   └── generator.py        # LLM generation + NLI faithfulness check
│
├── evaluation/
│   ├── bioasq_eval.py      # EM, F1, Recall@k, faithfulness rate
│   └── eda.py              # corpus analysis charts
│
└── diagrams/
    └── system_diagram.png  # system architecture diagram
```

---

## Data

Abstracts are downloaded automatically from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) via the [NLM E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25497/). No manual download required.

- Source: PubMed / NCBI NLM
- Scale: 10,000+ abstracts
- Topics: configurable via MeSH terms in `data/download_pubmed.py`
- Each chunk stores: PMID, passage text, position in abstract, year, MeSH tags

Evaluation uses the [BioASQ](http://bioasq.org/) dataset for question answering benchmarks.

---

## Evaluation

To run the full evaluation suite against BioASQ:

```bash
python evaluation/bioasq_eval.py
```

Outputs: Exact Match, F1, Recall@5, and faithfulness rate across retrieval strategies.

| Strategy | Recall@5 | F1 | Faithfulness |
|---|---|---|---|
| BM25 baseline | TBD | TBD | TBD |
| Dense retrieval (BioBERT) | TBD | TBD | TBD |
| Hybrid (BM25 + Dense) | TBD | TBD | TBD |
| Hybrid + Reranker | TBD | TBD | TBD |
| Hybrid + Reranker + Faithfulness filter | TBD | TBD | TBD |

*Results populated after ablation experiments.*

---

## Team

| Member | Role |
|---|---|
| A | Integration, infrastructure, Docker, GCP deployment |
| B | Data pipeline, retrieval (BM25, dense, hybrid) |
| C | Local LLM deployment, generation, faithfulness verification |
| D | Streamlit frontend, evaluation, EDA |

---

## Acknowledgements

- [PubMed / NLM E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/) — abstract corpus
- [BioASQ](http://bioasq.org/) — evaluation dataset
- [Ollama](https://ollama.com/) — local LLM serving
- [Qwen3](https://huggingface.co/Qwen) — base language model
- [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) — faithfulness NLI model
