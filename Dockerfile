FROM python:3.11-slim

# Install system dependencies + Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
