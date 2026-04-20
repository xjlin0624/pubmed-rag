#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    kill "${OLLAMA_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[entrypoint] Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

echo "[entrypoint] Waiting for Ollama API..."
until curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; do
  sleep 1
done
echo "[entrypoint] Ollama is ready."

MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
echo "[entrypoint] Ensuring model is pulled: ${MODEL}"
ollama pull "${MODEL}"

if [[ "${BUILD_RETRIEVAL_INDEX:-0}" == "1" ]]; then
  CHUNKS="${CHUNKS_FILE:-chunks.json}"
  INDEX="${INDEX_FILE:-faiss.index}"
  if [[ ! -f "${CHUNKS}" ]] || [[ ! -f "${INDEX}" ]]; then
    echo "[entrypoint] BUILD_RETRIEVAL_INDEX=1 — running retriever index build (needs network)..."
    python -m retriever.retriever || {
      echo "[entrypoint] Index build failed; pipeline will use mock retriever until indexes exist."
    }
  else
    echo "[entrypoint] Index files already present; skipping build."
  fi
fi

echo "[entrypoint] Launching Streamlit on 0.0.0.0:8501"
exec streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
