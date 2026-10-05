from fastapi import FastAPI
from pydantic import BaseModel
from .router.route import laya_router
from .graph.workflows import workflow

app = FastAPI()

class Query(BaseModel):
    user_message : str 

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def query(query:Query):
    route = laya_router({
        "user_message": query.user_message
    })

    result = workflow.invoke({
        "user_message": query.user_message,
        "route": route['intent'],
        "agent_response": "",
        "verified": False,
        "retry_count": 0,
        "verification_issues": [],
        "verification_reason": ""
    })

    return {
    "response": result["agent_response"],
    "agent": result["route"],
    "intent": route["intent"],
}