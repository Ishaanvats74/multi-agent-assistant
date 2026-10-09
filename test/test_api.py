
from unittest.mock import patch

from fastapi.testclient import TestClient

from multi_agent_assistant.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("multi_agent_assistant.main.workflow.invoke")
@patch("multi_agent_assistant.main.laya_router")
def test_chat_success(mock_router, mock_workflow):
    mock_router.return_value = {"intent": "technical"}

    mock_workflow.return_value = {
        "agent_response": "Binary search takes O(log n) time.",
        "route": "technical",
        "verified": True,
        "retry_count": 0,
        "verification_issues": [],
    }

    response = client.post("/chat",json={"query_input": "Explain binary search."})

    assert response.status_code == 200
    assert response.json()["intent"] == "technical"
    assert response.json()["response"] == ("Binary search takes O(log n) time.")
    mock_workflow.assert_called_once()


def test_chat_missing_message():
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_empty_message():
    response = client.post(
        "/chat",
        json={"query_input": "   "},
    )

    assert response.status_code == 422


@patch("multi_agent_assistant.main.laya_router")
def test_chat_rejects_routing_failure(mock_router):
    mock_router.side_effect = RuntimeError("internal provider details")

    response = client.post(
        "/chat",
        json={"query_input": "Explain binary search."},
    )

    assert response.status_code == 503
    assert "internal provider details" not in response.text


@patch("multi_agent_assistant.main.workflow.invoke")
@patch("multi_agent_assistant.main.laya_router")
def test_chat_handles_workflow_failure(mock_router, mock_workflow):
    mock_router.return_value = {"intent": "technical"}
    mock_workflow.side_effect = TimeoutError("internal timeout details")

    response = client.post("/chat",json={"query_input": "Explain binary search."})

    assert response.status_code == 500
    assert "internal timeout details" not in response.text
