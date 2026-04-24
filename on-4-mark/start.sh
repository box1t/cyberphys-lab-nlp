#!/bin/bash

set -e

echo "Starting Ollama..."
ollama serve &

# Ждём пока Ollama реально поднимется
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done

echo "Ollama is ready!"

# Загружаем модель
echo "Pulling model..."
ollama pull qwen2.5:0.5b

echo "Starting FastAPI..."
uvicorn app:app --host 0.0.0.0 --port 8000