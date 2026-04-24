import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

API_URL = "http://localhost:8000/analyze"


# =========================
# DATASET
# =========================

def load_dataset(path, limit=100):
    """
    Загружает и подготавливает датасет SMS Spam.

    Args:
        path (str): путь к csv
        limit (int): ограничение выборки

    Returns:
        list[dict]
    """
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df.head(limit).to_dict('records')


# =========================
# PROMPTS
# =========================

ZERO_SHOT = """
You are a spam classifier.
Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
0 = ham, 1 = spam.
"""

COT = """
You are a spam classifier.

Think step by step before answering.

Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
0 = ham, 1 = spam.
"""

FEW_SHOT = """
You are a spam classifier.

Examples:

SMS: "Win a free iPhone now!"
Output: {"reasoning": "Prize scam", "verdict": 1}

SMS: "Hey, are we meeting today?"
Output: {"reasoning": "Normal conversation", "verdict": 0}

SMS: "Call now to claim reward"
Output: {"reasoning": "Urgent scam", "verdict": 1}

Now classify:

Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
"""

COT_FEW_SHOT = """
You are a spam classifier.

Think step by step.

Examples:

SMS: "Win money now!"
Reasoning: Contains scam keywords and urgency.
Output: {"reasoning": "Scam message", "verdict": 1}

SMS: "Let's meet at 6"
Reasoning: Normal human communication.
Output: {"reasoning": "Normal message", "verdict": 0}

Now classify:

Respond ONLY JSON:
{"reasoning": "...", "verdict": 0 or 1}
"""


# =========================
# LLM QUERY
# =========================

def query_llm(text, prompt):
    """
    Отправляет SMS + prompt в LLM сервис.

    Returns:
        int | None
    """
    try:
        full_text = f"{prompt}\n\nSMS: {text}"

        res = requests.post(
            API_URL,
            json={"text": full_text},
            timeout=30
        )

        data = res.json()

        if "verdict" not in data:
            return None

        return int(data["verdict"])

    except Exception as e:
        print(f"Error: {e}")
        return None


# =========================
# EVALUATION
# =========================

def evaluate(dataset, prompt, name):
    """
    Оценивает технику промптинга.

    Returns:
        tuple
    """
    y_true = []
    y_pred = []

    print(f"\nRunning {name}...")

    for i, row in enumerate(dataset):
        pred = query_llm(row['text'], prompt)

        if pred is None:
            continue

        y_true.append(row['label'])
        y_pred.append(pred)

        if i % 20 == 0:
            print(f"Processed {i} samples")

        time.sleep(0.2)  # чтобы не перегружать модель

    if not y_true:
        print("No valid predictions!")
        return None

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    return acc, precision, recall, f1


# =========================
# MAIN EXPERIMENT
# =========================

def run_experiment():
    """
    Запускает исследование всех техник.
    """
    dataset = load_dataset("spam.csv", limit=100)

    techniques = {
        "zero-shot": ZERO_SHOT,
        "cot": COT,
        "few-shot": FEW_SHOT,
        "cot+few-shot": COT_FEW_SHOT
    }

    results = {}

    for name, prompt in techniques.items():
        metrics = evaluate(dataset, prompt, name)
        results[name] = metrics

    print("\n=== FINAL RESULTS ===")

    for name, metrics in results.items():
        if metrics:
            acc, p, r, f1 = metrics
            print(f"{name:15} | Acc={acc:.2f} | P={p:.2f} | R={r:.2f} | F1={f1:.2f}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_experiment()