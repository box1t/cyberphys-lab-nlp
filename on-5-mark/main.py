from fastapi import FastAPI
from pydantic import BaseModel
from mcp_client import route_request

app = FastAPI()


class MCPRequest(BaseModel):
    type: str
    age: int | None = None
    income: float | None = None
    has_job: bool | None = None
    education: bool | None = None
    married: bool | None = None


@app.get("/")
def root():
    return {"status": "MCP service running"}


@app.post("/process")
def process(req: MCPRequest):
    """
    Универсальный MCP endpoint
    """
    result = route_request(req.dict())
    return result

