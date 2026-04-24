import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000/analyze"


# ----------------------------
# 🔹 Загрузка Adult dataset
# ----------------------------
def load_data():
    """
    Загружает и очищает Adult dataset.
    """
    adult = fetch_ucirepo(id=2)

    X = adult.data.features
    y = adult.data.targets

    df = pd.concat([X, y], axis=1)

    # бинаризация таргета
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

    # обработка пропусков
    df = df.replace("?", None)
    df = df.dropna()

    return df.sample(1000, random_state=42)


# ----------------------------
# 🔹 Preprocessing (train)
# ----------------------------
def preprocess_train(df):
    """
    Кодирует категориальные признаки (fit).
    """
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


# ----------------------------
# 🔹 Preprocessing (test)
# ----------------------------
def preprocess_test(df, encoders):
    """
    Кодирует тестовые данные.
    Не падает на новых категориях.
    """
    df = df.copy()

    for col in df.columns:
        if col in encoders:
            le = encoders[col]

            # заменяем unseen категории на -1
            df[col] = df[col].apply(
                lambda x: x if x in le.classes_ else "__unknown__"
            )

            # добавляем __unknown__ в encoder
            if "__unknown__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__unknown__")

            df[col] = le.transform(df[col])

    return df


# ----------------------------
# 🔹 LLM prediction
# ----------------------------
def llm_predict(row):
    """
    LLM предсказывает доход по признакам.
    Используются RAW (не закодированные) данные.
    """

    text = f"""
    You are a credit risk classifier.

    Person:
    age={row.get('age')}
    education={row.get('education')}
    hours_per_week={row.get('hours-per-week')}
    occupation={row.get('occupation')}

    Task:
    Predict if income is >50K.

    Respond ONLY JSON:
    {{"verdict": 0 or 1}}
    """

    try:
        res = requests.post(API_URL, json={"text": text}, timeout=5)
        return int(res.json().get("verdict", 0))
    except Exception:
        return 0


# ----------------------------
# 🔹 Hybrid (MCP логика)
# ----------------------------
def hybrid_predict(row_ml, row_raw, rf_model):
    """
    Комбинирует ML и LLM.
    """
    proba = rf_model.predict_proba(row_ml.to_frame().T)[0][1]

    if proba > 0.7:
        return 1
    if proba < 0.3:
        return 0

    return llm_predict(row_raw)


# ----------------------------
# 🔹 Метрики
# ----------------------------
def evaluate(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 2),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 2),
    }

# ----------------------------
# 🔹 LLM batch (параллельно)
# ----------------------------
def batch_llm_predict(rows, max_workers=5):
    """
    Параллельные запросы к LLM (ускорение x3–x5)
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(llm_predict, row) for row in rows]

        for i, future in enumerate(as_completed(futures)):
            try:
                results.append(future.result())
            except:
                results.append(0)

            # 🔥 прогресс
            if i % 5 == 0:
                print(f"[LLM] processed {i}/{len(rows)}")

    return results


# ----------------------------
# 🔹 Основной эксперимент (FIXED)
# ----------------------------
def run():
    df_raw = load_data()

    train_raw, test_raw = train_test_split(df_raw, test_size=0.3, random_state=42)

    train, encoders = preprocess_train(train_raw)
    test = preprocess_test(test_raw, encoders)

    y_train = train["income"]
    y_test = test["income"]

    X_train = train.drop(columns=["income"])
    X_test = test.drop(columns=["income"])

    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # модели
    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(n_estimators=100)

    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)

    # ML predictions
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test)

    # 🔥 ОГРАНИЧЕНИЕ (важно)
    MAX_SAMPLES = 100

    X_test_small = X_test.head(MAX_SAMPLES)
    y_test_small = y_test.head(MAX_SAMPLES)
    test_raw_small = test_raw.head(MAX_SAMPLES)

    print(f"\nRunning evaluation on {MAX_SAMPLES} samples...\n")

    # ----------------------------
    # 🔥 LLM (параллельно)
    # ----------------------------
    start = time.time()

    y_pred_llm = batch_llm_predict(
        [row for _, row in test_raw_small.iterrows()],
        max_workers=5
    )

    print(f"LLM done in {time.time() - start:.1f}s\n")

    # ----------------------------
    # 🔥 Hybrid (быстро + fallback LLM)
    # ----------------------------
    y_pred_hybrid = []

    start = time.time()

    for i, ((_, row_ml), (_, row_raw)) in enumerate(zip(X_test_small.iterrows(), test_raw_small.iterrows())):

        if i % 10 == 0:
            print(f"[Hybrid] {i}/{MAX_SAMPLES}")

        proba = rf.predict_proba(row_ml.to_frame().T)[0][1]

        if proba > 0.8:
            y_pred_hybrid.append(1)
        elif proba < 0.2:
            y_pred_hybrid.append(0)
        else:
            y_pred_hybrid.append(llm_predict(row_raw))

    print(f"Hybrid done in {time.time() - start:.1f}s\n")

    # ----------------------------
    # 🔹 RESULTS
    # ----------------------------
    print("\n=== FINAL RESULTS ===")

    print("LLM:", evaluate(y_test_small, y_pred_llm))
    print("Logistic:", evaluate(y_test, y_pred_lr))
    print("RandomForest:", evaluate(y_test, y_pred_rf))
    print("Hybrid:", evaluate(y_test_small, y_pred_hybrid))

# ----------------------------
if __name__ == "__main__":
    run()