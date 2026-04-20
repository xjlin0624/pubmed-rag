# CLAUDE.md — PubMed RAG Project Context

## What this project is

A retrieval-augmented generation (RAG) system for answering medical questions grounded in PubMed abstracts. Built for CS 6120 NLP, Northeastern University.

Pipeline: `User query → Hybrid Retriever (BM25 + Dense) → Local LLM (Ollama) → Faithfulness Check → Cited Answer`

## Team roles and module ownership

| Member | Role | Owns |
|---|---|---|
| A | Integration, infrastructure, Docker, GCP deployment | `Dockerfile`, `entrypoint.sh`, deployment config |
| B | Data pipeline, retrieval | `data/`, `retriever/` |
| C | Local LLM deployment, generation, faithfulness verification | `generator/`, `pipeline.py` |
| D | Streamlit frontend, evaluation, EDA | `app.py`, `evaluation/` |

## Retriever interface contract

B must implement `retriever/retriever.py` with:

```python
def retrieve(query: str, top_k: int) -> list[dict]:
    ...
```

Each returned dict: `{"pmid": str, "text": str, "score": float}`

`pipeline.py` lazy-imports this when real index files (`chunks.json` + `faiss.index`) are present; otherwise it falls back to a mock retriever so other members can test end-to-end without the indexes built.

## Key design decisions

- **LLM**: Qwen2.5:7b served locally via Ollama — no external API calls
- **Faithfulness check**: NLI entailment using `cross-encoder/nli-MiniLM2-L6-H768`, sentence-level
- **Retrieval**: Hybrid BM25 + MedCPT dense retrieval, FAISS index; cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
- **Embedding**: MedCPT (ncbi/MedCPT-Article-Encoder) — domain-specific biomedical embeddings
- **Evaluation dataset**: PubMedQA (1k expert-annotated QA pairs); diabetes-filtered subset (28 Q) for in-domain eval
- **Ablation order**: BM25 → Dense → Hybrid → +Reranker → +Faithfulness filter

## Running locally (without Docker)

```bash
cp .env.example .env
pip install -r requirements.txt
ollama pull qwen2.5:7b && ollama serve
python pipeline.py   # end-to-end test with mock retriever
```

## Environment variables

See `.env.example`. All have sensible defaults — no changes needed for local dev.
