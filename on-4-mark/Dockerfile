FROM python:3.12

# Установка системных пакетов
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Установка Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY app.py .
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 11434

CMD ["./start.sh"]