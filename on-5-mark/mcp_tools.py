from ml_tools import ml_predict

import requests

def llm_credit_tool(text: str):
    """
    MCP tool: оценка через LLM.
    """
    res = requests.post(
        "http://localhost:8000/analyze",
        json={"text": text}
    )

    return res.json()

def ml_credit_assessment(data):
    """
    Упрощённый ML tool (без сложного pipeline).
    """

    score = 0

    if data.get("education"):
        score += 1

    if data.get("married"):
        score += 1

    decision = "APPROVE" if score >= 1 else "REJECT"

    return {
        "ml_score": score,
        "decision": decision,
        "source": "ML_simple"
    }

def calculate_credit_score(data):
    income = data.get("income", 0)
    age = data.get("age", 0)
    history = data.get("credit_history", 0)

    score = 0

    if income > 40000:
        score += 40

    if age > 25:
        score += 20

    if history > 1:
        score += 30

    decision = "APPROVE" if score >= 50 else "REJECT"

    return {
        "credit_score": score,
        "decision": decision
    }


def assess_risk(data):
    education = data.get("education", False)
    married = data.get("married", False)

    score = 0

    if not education:
        score += 0.5

    if not married:
        score += 0.3

    risk = "HIGH_RISK" if score > 0.5 else "LOW_RISK"

    return {
        "risk_score": score,
        "decision": risk
    }
