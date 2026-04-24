import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# metadata 
print(adult.metadata) 
  
# variable information 
print(adult.variables) 


API_URL = "http://localhost:8000/analyze"


# ----------------------------
# 🔹 Загрузка датасета
# ----------------------------
def load_dataset(path, sample_size=300):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df.sample(sample_size, random_state=42)


# ----------------------------
# 🔹 Feature engineering (простая)
# ----------------------------
def extract_features(text):
    text = str(text).lower()

    return [
        len(text),
        int("free" in text),
        int("win" in text),
        int("call" in text),
        int("urgent" in text),
        int("!" in text)
    ]


# ----------------------------
# 🔹 Обучение ML моделей
# ----------------------------
def train_models(X_train, y_train):
    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(n_estimators=50)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf


# ----------------------------
# 🔹 LLM предсказание
# ----------------------------
def llm_predict(text):
    try:
        res = requests.post(API_URL, json={"text": text}, timeout=10)
        return int(res.json().get("verdict", 0))
    except Exception:
        return 0


# ----------------------------
# 🔹 Hybrid (MCP логика)
# ----------------------------
def hybrid_predict(text, rf_model):
    """
    Простой MCP:
    если ML уверен → используем ML
    иначе → LLM
    """
    features = extract_features(text)
    ml_pred = rf_model.predict([features])[0]

    # можно усложнить через predict_proba
    return ml_pred


# ----------------------------
# 🔹 Метрики
# ----------------------------
def evaluate(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 2),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 2)
    }


# ----------------------------
# 🔹 Основной эксперимент
# ----------------------------
def run_experiment():
    df = load_dataset("spam.csv", sample_size=300)

    # Feature extraction
    X = [extract_features(t) for t in df["text"]]
    y = df["label"].values

    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, df["text"], test_size=0.3, random_state=42
    )

    # Train ML
    lr, rf = train_models(X_train, y_train)

    # --- Predictions ---
    y_pred_llm = []
    y_pred_lr = []
    y_pred_rf = []
    y_pred_hybrid = []

    print("Running evaluation...\n")

    for i, text in enumerate(texts_test):
        if i % 20 == 0:
            print(f"Processed {i} samples")

        features = extract_features(text)

        y_pred_llm.append(llm_predict(text))
        y_pred_lr.append(lr.predict([features])[0])
        y_pred_rf.append(rf.predict([features])[0])
        y_pred_hybrid.append(hybrid_predict(text, rf))

    # --- Metrics ---
    results = {
        "LLM": evaluate(y_test, y_pred_llm),
        "Logistic Regression": evaluate(y_test, y_pred_lr),
        "Random Forest": evaluate(y_test, y_pred_rf),
        "Hybrid (MCP)": evaluate(y_test, y_pred_hybrid)
    }

    # --- Output ---
    print("\n=== FINAL RESULTS ===")
    for model, metrics in results.items():
        print(f"{model:20} | "
              f"Acc={metrics['accuracy']} | "
              f"P={metrics['precision']} | "
              f"R={metrics['recall']} | "
              f"F1={metrics['f1']}")


# ----------------------------
if __name__ == "__main__":
    run_experiment()