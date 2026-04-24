import requests
from sklearn.metrics import classification_report, accuracy_score


def run_evaluation(dataset):
    """
    Оценивает модель.

    Args:
        dataset (list): [{'text': str, 'label': int}]
    """
    y_true = []
    y_pred = []

    print("Evaluating...")

    for entry in dataset:
        try:
            res = requests.post(
                "http://localhost:8000/analyze",
                json={"text": entry['text']},
                timeout=15
            )

            data = res.json()

            # 🔥 ВАЖНО: проверка ответа
            if "verdict" not in data or data["verdict"] is None:
                print(f"Invalid response: {data}")
                continue

            y_true.append(entry['label'])
            y_pred.append(int(data["verdict"]))

        except Exception as e:
            print(f"Skip error: {e}")

    # 🔥 защита от пустых данных
    if not y_true:
        print("No valid predictions")
        return

    print("\n--- PERFORMANCE REPORT ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(classification_report(y_true, y_pred))

test_data = [
    {"text": "Your loan is approved! Call now!", "label": 1},
    {"text": "Hey, are we meeting today?", "label": 0},
    {"text": "WIN A FREE IPHONE!!! CLICK NOW", "label": 1},
]


if __name__ == "__main__":
    run_evaluation(test_data)