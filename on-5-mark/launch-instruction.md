
```sh
# сборка образа:
# cd on-5-mark
```

```sh
docker-compose build --no-cache # сборка образа (633 сек)
```
# Шаг-1

```sh
docker-compose down 
docker-compose up --build -d 
docker logs on-5-mark_mcp-service_1
```

# Шаг-2
```sh 
curl http://localhost:8000/ # health-check endpoint
```

# Шаг-3
```sh
# Скрипты вне контейнера, отправляющие запросы (curl)

# тест-1: кредитный скор
curl -X POST http://localhost:8001/process \
-H "Content-Type: application/json" \
-d '{
  "type": "credit",
  "income": 50000,
  "age": 30,
  "credit_history": 2
}'

# тест-2: риск
curl -X POST http://localhost:8001/process \
-H "Content-Type: application/json" \
-d '{
  "type": "risk",
  "education": true,
  "married": false
}'
```
