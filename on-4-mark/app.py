import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"


# 🔥 Все техники промптинга
PROMPTS = {
    "zero-shot": """
You are an SMS spam classifier.

Classify message:
0 = Ham
1 = Spam

Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
""",

    "cot": """
You are an SMS spam classifier.

Think step by step:
1. Analyze message
2. Detect suspicious patterns
3. Decide spam or not

Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
""",

    "few-shot": """
You are an SMS spam classifier.

Examples:
SMS: "Win a free iPhone now!"
Output: {"reasoning": "Contains prize scam", "verdict": 1}

SMS: "Hey, are we meeting today?"
Output: {"reasoning": "Normal conversation", "verdict": 0}

Now classify.

Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
""",

    "cot+few-shot": """
You are an SMS spam classifier.

Think step by step:
1. Analyze message
2. Detect spam patterns
3. Make decision

Examples:
SMS: "Win a free iPhone now!"
Output: {"reasoning": "Contains prize scam", "verdict": 1}

SMS: "Hey, are we meeting today?"
Output: {"reasoning": "Normal conversation", "verdict": 0}

Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
"""
}


class SMSRequest(BaseModel):
    text: str
    mode: Optional[str] = "zero-shot"   # 🔥 выбор техники
    prompt: Optional[str] = None        # 🔥 кастомный prompt


def extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                return json.loads(text[start:end])
            except Exception:
                return None
    return None


def normalize_verdict(verdict):
    if verdict in [0, "0", "ham", "Ham"]:
        return 0
    if verdict in [1, "1", "spam", "Spam"]:
        return 1
    return 0


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: SMSRequest):
    """
    Универсальный endpoint:

    - mode: zero-shot | cot | few-shot | cot+few-shot
    - prompt: кастомный prompt (перекрывает mode)

    Returns:
        dict: reasoning + verdict
    """

    # 🔥 приоритет: кастомный prompt > mode
    if request.prompt:
        system_prompt = request.prompt
    else:
        system_prompt = PROMPTS.get(request.mode, PROMPTS["zero-shot"])

    payload = {
        "model": "qwen2.5:0.5b",
        "prompt": f"{system_prompt}\n\nSMS: {request.text}",
        "stream": False,
        "format": "json",
        "temperature": 0.0,
        "num_predict": 30
    }

    # 🔹 запрос к Ollama
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"reasoning": f"Ollama error: {e}", "verdict": 0}

    # 🔹 JSON от Ollama
    try:
        data = response.json()
    except Exception:
        return {"reasoning": "Invalid Ollama response", "verdict": 0}

    raw_content = data.get("response", "")

    parsed = extract_json(raw_content)

    if not parsed:
        return {"reasoning": raw_content, "verdict": 0}

    reasoning = str(parsed.get("reasoning", raw_content))
    verdict = normalize_verdict(parsed.get("verdict"))

    return {
        "reasoning": reasoning,
        "verdict": verdict
    }