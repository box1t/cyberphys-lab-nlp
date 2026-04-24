import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class CreditMLPipeline:
    """
    ML pipeline для кредитного скоринга (Adult dataset).
    """

    def __init__(self):
        self.encoders = {}
        self.lr = LogisticRegression(max_iter=200)
        self.rf = RandomForestClassifier(n_estimators=100)

    def preprocess(self, df: pd.DataFrame, fit=False):
        """
        Кодирует категориальные признаки.
        """
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == "object":
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    df[col] = le.transform(df[col])

        return df

    def train(self, df: pd.DataFrame):
        """
        Обучение моделей.
        """
        df = df.copy()

        y = df["income"]
        X = df.drop(columns=["income"])

        X = self.preprocess(X, fit=True)

        self.lr.fit(X, y)
        self.rf.fit(X, y)

    def predict(self, data: dict):
        """
        Предсказание для одного клиента.
        """
        df = pd.DataFrame([data])
        df = self.preprocess(df)

        lr_pred = int(self.lr.predict(df)[0])
        rf_pred = int(self.rf.predict(df)[0])

        return {
            "logistic": lr_pred,
            "random_forest": rf_pred
        }
    
def ml_predict(data):
    """
    Простейший ML tool для MCP.

    Args:
        data (dict): входные данные

    Returns:
        dict: результат ML скоринга
    """
    score = 0

    if data.get("education"):
        score += 1

    if data.get("married"):
        score += 1

    decision = "APPROVE" if score >= 1 else "REJECT"

    return {
        "ml_score": score,
        "decision": decision
    }