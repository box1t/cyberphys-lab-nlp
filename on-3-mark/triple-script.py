import requests
import json
import csv

def send_prompt_to_ollama(prompt: str, model: str = "qwen2.5:0.5b") -> str:
    """
    Отправляет HTTP POST запрос к API Ollama для генерации ответа.

    Args:
        prompt (str): Текст запроса к модели.
        model (str): Название модели на сервере. По умолчанию "qwen2.5:0.5b".

    Returns:
        str: Ответ от модели. Если произошла ошибка, возвращает описание ошибки.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Ошибка запроса: {e}"

def run_inference_test(prompts: list) -> list:
    """
    Запускает цикл инференса для списка запросов.

    Args:
        prompts (list): Список строк с вопросами для LLM.

    Returns:
        list: Список кортежей вида [(запрос, ответ), ...].
    """
    results = []
    print(f"Запуск инференса для {len(prompts)} запросов...")
    
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/10] Обработка: {p[:30]}...")
        answer = send_prompt_to_ollama(p)
        results.append((p, answer))
    
    return results

def save_report(results: list, filename: str = "inference_report.csv"):
    """
    Сохраняет результаты инференса в CSV файл (отчет).

    Args:
        results (list): Результаты инференса.
        filename (str): Имя файла для сохранения.
    """
    with open(filename, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Запрос к LLM", "Вывод LLM"])  # Заголовки столбцов
        writer.writerows(results)
    print(f"Отчет успешно сохранен в файл: {filename}")

if __name__ == "__main__":
    # 10 произвольных запросов
    test_queries = [
        "What is the capital of Kazakhstan?",
        "Explain quantum entanglement in one sentence.",
        "Write a Python function to sort a list.",
        "Who wrote 'Crime and Punishment'?",
        "How many planets are in the solar system?",
        "Translate 'Artificial Intelligence' to French.",
        "What is the chemical formula for water?",
        "Give me a short recipe for pancakes.",
        "What is the primary function of a CPU?",
        "Write a 4-line poem about the moon."
    ]

    # Выполнение задач
    inference_data = run_inference_test(test_queries)
    save_report(inference_data)