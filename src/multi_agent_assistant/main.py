import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from .observability.metrics import RequestMetrics,ModelMetricsCallback
from .router.route import laya_router
from .graph.workflows import workflow

app = FastAPI()
logger = logging.getLogger(__name__)

ALLOWED_INTENTS = {"technical", "planning", "financial"}


class Query(BaseModel):
    query_input: str = Field(min_length=1, max_length=5000)

    @field_validator("query_input")
    @classmethod
    def validate_query_input(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("query_input cannot be empty.")

        return value


@app.middleware("http")
async def collect_request_metrics(request: Request, call_next):
    metrics = RequestMetrics()
    request.state.metrics = metrics

    try:
        response = await call_next(request)
        return response
    finally:
        logger.info(
            "http_request_metrics",
            extra={
                "method": request.method,
                "path": request.url.path,
                "metrics": metrics.to_dict(),
            },
        )


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def query(query: Query, request: Request):

    try:
        metrics: RequestMetrics = request.state.metrics
        metrics.laya_calls += 1
        started = metrics.start_stage()

        route = laya_router({"query_input": query.query_input})

        metrics.end_stage("laya_routing", started)

    except Exception:
        logger.exception("Laya routing failed")

        raise HTTPException(status_code=503,detail="Intent routing is temporarily unavailable.")

    if not isinstance(route, dict) or route.get("intent") not in ALLOWED_INTENTS:
        logger.error("Laya returned an invalid intent response")

        raise HTTPException(status_code=503,detail="Unable to determine the request intent.")

    try:
        started = metrics.start_stage()
        result = workflow.invoke({
                "query_input": query.query_input,
                "route": route["intent"],
                "agent_response": "",
                "verified": False,
                "retry_count": 0,
                "verification_issues": [],
                "verification_reason": ""
            },
            config={"recursion_limit": 10, "callbacks": [ModelMetricsCallback(metrics)]}
        )
        metrics.end_stage("workflow", started)

    except Exception:
        logger.exception("Workflow execution failed")

        raise HTTPException(status_code=500,detail="An internal error occurred while processing your request.")


    if not isinstance(result, dict):
        logger.error("Workflow returned an invalid result")

        raise HTTPException(status_code=500,detail="The workflow returned an invalid response.")

    if not isinstance(result.get("agent_response"), str):
        logger.error("Workflow response is missing or invalid")

        raise HTTPException(status_code=500,detail="Unable to generate a valid response.")

    metrics.retries = result.get("retry_count", 0)
    metrics.verification_failures = metrics.retries if not result.get("verified", False) else 0

    return {
        "response": result["agent_response"],
        "agent": result["route"],
        "intent": route["intent"],
        "verified": result.get("verified", False),
        "retry_count": metrics.retries,
    }
