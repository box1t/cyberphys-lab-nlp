from mcp_tools import calculate_credit_score, assess_risk


# def route_request(data: dict):
#     """
#     Простейший MCP-клиент:
#     выбирает tool по полю 'type'
#     """

#     request_type = data.get("type")

#     if request_type == "credit":
#         return calculate_credit_score(
#             age=data.get("age", 0),
#             income=data.get("income", 0),
#             has_job=data.get("has_job", False)
#         )

#     elif request_type == "risk":
#         return assess_risk(
#             education=data.get("education", False),
#             married=data.get("married", False)
#         )

#     else:
#         return {"error": "Unknown request type"}

from mcp_tools import ml_credit_assessment, llm_credit_tool

def ml_predict(data):
    """
    Простейший ML-скоринг (заглушка или модель).
    """
    score = 0

    if data.get("education"):
        score += 1

    if data.get("married"):
        score += 1

    return {
        "ml_score": score,
        "decision": "APPROVE" if score >= 1 else "REJECT"
    }
def route_request(data):
    t = data.get("type")

    if t == "credit":
        return calculate_credit_score(data)

    elif t == "risk":
        return assess_risk(data)

    elif t == "ml":
        return ml_credit_assessment(data)

    elif t == "hybrid":
        return hybrid_decision(data)

    else:
        return {"error": f"Unknown type: {t}"}


# def hybrid_decision(data: dict):
#     """
#     Hybrid MCP логика:
#     ML + LLM
#     """

#     ml_result = ml_credit_assessment(data)
#     llm_result = llm_credit_tool(data.get("text", ""))

#     # 🔥 простая стратегия
#     rf_pred = ml_result["result"]["random_forest"]

#     if rf_pred == 1:
#         return {
#             "final": 1,
#             "source": "ML",
#             "details": ml_result
#         }

#     return {
#         "final": llm_result.get("verdict", 0),
#         "source": "LLM",
#         "details": llm_result
#     }

def hybrid_decision(data):
    """
    MCP logic:
    ML → baseline
    LLM → fallback / explanation
    """
    ml_res = ml_predict(data)

    if ml_res["ml_score"] >= 1:
        return ml_res
    else:
        return {
            "decision": "REJECT",
            "source": "ML"
        }