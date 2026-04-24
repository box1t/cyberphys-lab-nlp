
```sh
# сборка образа:
# cd on-4-mark
# 4. docker build -t sms-spam-llm . 

```

```sh
docker-compose build --no-cache # сборка образа (633 сек)
```

```sh
docker-compose down 
docker-compose up --build -d 
docker logs -f spam-detector # ожидать +- 60 сек для скачивания модели
```

```sh 
curl http://localhost:8000/ # health-check endpoint


# Скрипт вне контейнера, отправляющий запросы (curl)
curl -X POST http://localhost:8000/analyze \
-H "Content-Type: application/json" \
-d '{"text": "WIN FREE MONEY NOW!!!"}'
```


```sh
# Скрипты вне контейнера, отправляющие запросы (curl)

# 🔹 1. Zero-shot
curl -X POST http://localhost:8000/analyze \
-H "Content-Type: application/json" \
-d '{"text": "WIN FREE MONEY NOW!!!", "mode": "zero-shot"}'

# 🔹 2. CoT
curl -X POST http://localhost:8000/analyze \
-H "Content-Type: application/json" \
-d '{"text": "WIN FREE MONEY NOW!!!", "mode": "cot"}'

# 🔹 3. Few-shot
curl -X POST http://localhost:8000/analyze \
-H "Content-Type: application/json" \
-d '{"text": "WIN FREE MONEY NOW!!!", "mode": "few-shot"}'


# 🔹 4. CoT + Few-shot
curl -X POST http://localhost:8000/analyze \
-H "Content-Type: application/json" \
-d '{"text": "WIN FREE MONEY NOW!!!", "mode": "cot+few-shot"}'


```

```sh
python3 smoke_test.py

python3 research_experiment.py

```