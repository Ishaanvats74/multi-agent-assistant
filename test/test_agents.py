from unittest.mock import MagicMock
from multi_agent_assistant.graph.workflows import technical_node,planning_node,finance_node


def make_state():
    return {
        "query_input": "Create a test response",
        "route": "planning",
        "agent_response": "",
        "verified": False,
        "verification_issues": [],
        "verification_reason": "",
        "retry_count": 0,
    }


def test_planning_node(monkeypatch):

    fake_agent = MagicMock()

    fake_agent.invoke.return_value = {
        "messages": [
            MagicMock(
                content="This is a planning response."
            )
        ]
    }

    monkeypatch.setattr(
        "multi_agent_assistant.graph.workflows.planning_agent",
        fake_agent,
    )

    state = make_state()

    result = planning_node(state)

    assert result["agent_response"] == (
        "This is a planning response."
    )

    fake_agent.invoke.assert_called_once()


def test_finance_node(monkeypatch):

    fake_agent = MagicMock()

    fake_agent.invoke.return_value = {
        "messages": [
            MagicMock(
                content="This is a finance response."
            )
        ]
    }

    monkeypatch.setattr("multi_agent_assistant.graph.workflows.finance_agent",fake_agent)

    state = make_state()

    result = finance_node(state)

    assert result["agent_response"] == "This is a finance response."

    fake_agent.invoke.assert_called_once()


def test_technical_node(monkeypatch):

    fake_agent = MagicMock()

    fake_agent.invoke.return_value = {
        "messages": [
            MagicMock(
                content="This is a technical response."
            )
        ]
    }

    monkeypatch.setattr("multi_agent_assistant.graph.workflows.technical_agent",fake_agent)

    state = make_state()

    result = technical_node(state)

    assert result["agent_response"] == "This is a technical response."

    fake_agent.invoke.assert_called_once()


def test_planning_retry_contains_verification_feedback(monkeypatch):

    fake_agent = MagicMock()

    fake_agent.invoke.return_value = {
        "messages": [
            MagicMock(
                content="Improved planning response."
            )
        ]
    }

    monkeypatch.setattr("multi_agent_assistant.graph.workflows.planning_agent",fake_agent)

    state = {
    "query_input": "Create a DSA plan",
    "route": "planning",
    "agent_response": "Bad previous response",
    "verified": False,
    "verification_issues": [
        "The plan exceeds 30 minutes per day."
    ],
    "verification_reason": "The previous response violated the time constraint.",
    "retry_count": 1,
}

    result = planning_node(state)

    assert result["agent_response"] == "Improved planning response."

    call_args = fake_agent.invoke.call_args

    prompt = call_args.args[0]["messages"][0]["content"]

    assert "30 minutes per day" in prompt
    assert "Bad previous response" in prompt