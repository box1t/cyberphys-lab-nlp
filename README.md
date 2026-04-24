Вот тебе **идеальный README уровня 5+**, уже структурированный, аккуратный и “защитный” — просто вставь в `README.md` 👇

---

# 📱 LLM + MCP для задач спам-фильтрации и кредитного скоринга

## 📚 Оглавление

1. [Описание проекта](#-описание-проекта)
2. [Архитектура](#-архитектура)
3. [Технологический стек](#-технологический-стек)
4. [Часть 1: LLM сервис (SMS spam)](#-часть-1-llm-сервис-sms-spam)
5. [Часть 2: MCP сервис (credit scoring)](#-часть-2-mcp-сервис-credit-scoring)
6. [Исследование prompt engineering](#-исследование-prompt-engineering)
7. [ML + Hybrid исследование (5+)](#-ml--hybrid-исследование-5)
8. [Результаты](#-результаты)
9. [Выводы](#-выводы)
10. [Как запустить](#-как-запустить)
11. [Тестирование](#-тестирование)

---

# 📌 Описание проекта

Проект представляет собой **proof-of-concept систему**, демонстрирующую применение:

* LLM (Qwen2.5:0.5B через Ollama)
* FastAPI сервисов
* MCP (Model Control Plane)
* классических ML моделей

для решения задач:

* 📩 фильтрации SMS спама
* 💳 кредитного скоринга

---

# 🏗 Архитектура

```text
            ┌──────────────┐
            │   Client     │
            └──────┬───────┘
                   │
        ┌──────────▼──────────┐
        │    FastAPI (LLM)     │  ← /analyze
        └──────────┬──────────┘
                   │
             ┌─────▼─────┐
             │  Ollama   │
             │ Qwen2.5   │
             └───────────┘


        ┌────────────────────────┐
        │     MCP Service        │  ← /process
        └──────────┬────────────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
┌────▼────┐  ┌─────▼────┐  ┌────▼────┐
│ Credit  │  │  Risk    │  │   ML     │
│ Tool    │  │ Tool     │  │ Tool     │
└─────────┘  └──────────┘  └─────────┘
                   │
              ┌────▼────┐
              │ Hybrid  │
              └─────────┘
```

---

# 🛠 Технологический стек

* Python 3.10+
* Docker / Docker Compose
* FastAPI
* Ollama
* Qwen2.5:0.5B
* scikit-learn
* pandas
* UCI Adult Dataset

---

# 🤖 Часть 1: LLM сервис (SMS spam)

## 📌 Реализация

* FastAPI endpoint:

```bash
POST /analyze
```

* LLM классифицирует SMS как:

```json
{"verdict": 0 или 1}
```

---

## ✅ Пример запроса

```bash
curl -X POST http://localhost:8000/analyze \
-H "Content-Type: application/json" \
-d '{"text": "WIN FREE MONEY NOW!!!"}'
```

## 📥 Ответ

```json
{
  "reasoning": "Spam message",
  "verdict": 1
}
```

---

# 🧠 Часть 2: MCP сервис (credit scoring)

## 📌 Реализация

Endpoint:

```bash
POST /process
```

---

## 🔹 Доступные инструменты

### 1. Credit Score

```json
{
  "type": "credit",
  "income": 50000,
  "age": 30,
  "credit_history": 2
}
```

---

### 2. Risk Assessment

```json
{
  "type": "risk",
  "education": true,
  "married": false
}
```

---

### 3. ML Tool

```json
{
  "type": "ml",
  "education": true,
  "married": true
}
```

---

### 4. Hybrid (MCP)

```json
{
  "type": "hybrid",
  "education": true,
  "married": false
}
```

---

# 🧪 Исследование prompt engineering

Проведено сравнение техник:

* zero-shot
* Chain-of-Thought (CoT)
* few-shot
* CoT + few-shot

---

## 📊 Результаты

| Метод          | Accuracy | Precision | Recall | F1       |
| -------------- | -------- | --------- | ------ | -------- |
| zero-shot      | 0.69     | 0.27      | 0.47   | 0.34     |
| CoT            | 0.62     | 0.23      | 0.53   | 0.32     |
| few-shot       | 0.75     | 0.36      | 0.59   | 0.44     |
| CoT + few-shot | **0.77** | **0.38**  | 0.53   | **0.44** |

---

## 📌 Вывод

```text
Few-shot значительно улучшает качество LLM.
```

---

# 📊 ML + Hybrid исследование (5+)

## 📌 Датасет

Adult Income Dataset (UCI)

Задача:

```text
income > 50K → 1
income ≤ 50K → 0
```

---

## 🔹 Подходы

* LLM
* Logistic Regression
* Random Forest
* Hybrid (MCP)

---

# 📈 Результаты

```text
LLM:            Acc=0.78 | P=0.00 | R=0.00 | F1=0.00
Logistic:       Acc=0.83 | P=0.72 | R=0.52 | F1=0.60
RandomForest:   Acc=0.83 | P=0.70 | R=0.59 | F1=0.64
Hybrid (MCP):   Acc=0.88 | P=0.85 | R=0.52 | F1=0.65
```

---

# 📉 Анализ

## LLM

* высокая accuracy за счёт дисбаланса
* не предсказывает класс 1

## ML

* стабильные результаты
* лучшее обобщение

## Hybrid

* лучший общий результат
* использует сильные стороны обоих подходов

---

# 🧾 Выводы

### 🔹 1. LLM неэффективен как standalone

```text
LLM плохо работает с табличными данными
```

---

### 🔹 2. ML превосходит LLM

```text
RandomForest > Logistic > LLM
```

---

### 🔹 3. Hybrid — лучший

```text
Hybrid > ML > LLM
```

---

## 🎯 Финальный вывод

```text
Наиболее эффективной является гибридная MCP-архитектура,
объединяющая LLM и классические ML модели.
```

---

# 🚀 Как запустить

## 1. Поднять LLM сервис

```bash
docker-compose up -d
```

---

## 2. Проверка

```bash
curl http://localhost:8000/
```

---

## 3. MCP сервис

```bash
docker-compose up -d
```

---

# 🧪 Тестирование

## LLM

```bash
curl -X POST http://localhost:8000/analyze \
-d '{"text": "Win money now!"}'
```

---

## MCP

```bash
curl -X POST http://localhost:8001/process \
-d '{"type": "hybrid", "education": true, "married": false}'
```

---

# 🏁 Итог

```text
Проект демонстрирует практическую применимость LLM,
но подтверждает, что максимальная эффективность достигается
в гибридной MCP-архитектуре.
```

---

## 💬 Если защищаешься

Главная мысль:

```text
Я показал, что LLM сам по себе слаб,
но в комбинации с ML даёт лучший результат.
```

---