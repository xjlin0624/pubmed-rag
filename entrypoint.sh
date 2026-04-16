#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434 > /dev/null; do
    sleep 1
done
echo "Ollama is ready."

# Pull the model if not already present
MODEL=${OLLAMA_MODEL:-qwen3:4b}
echo "Pulling model: $MODEL"
ollama pull "$MODEL"

# Launch the app
echo "Starting app..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Keep container alive; shut down Ollama on exit
wait $OLLAMA_PID
