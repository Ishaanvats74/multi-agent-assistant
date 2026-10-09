
from types import SimpleNamespace
from unittest.mock import patch

from multi_agent_assistant.graph.workflows import workflow


def initial_state():
    return {
        "user_message": "Explain binary search.",
        "route": "technical",
        "agent_response": "",
        "verified": False,
        "retry_count": 0,
        "verification_issues": [],
        "verification_reason": "",
    }


def test_rejected_answer_is_retried_and_then_approved():
    responses = [
        {"messages": [SimpleNamespace(content="First answer")]},
        {"messages": [SimpleNamespace(content="Improved answer")]},
    ]

    verification_results = [
        SimpleNamespace(
            approved=False,
            issues=["Missing complexity analysis"],
            reason="Incomplete answer",
        ),
        SimpleNamespace(
            approved=True,
            issues=[],
            reason="Sufficient answer",
        ),
    ]

    with (
        patch(
            "multi_agent_assistant.graph.workflows."
            "technical_agent.invoke",
            side_effect=responses,
        ) as mock_agent,
        patch(
            "multi_agent_assistant.graph.workflows.verify_response",
            side_effect=verification_results,
        ),
    ):
        result = workflow.invoke(initial_state())

    assert mock_agent.call_count == 2
    assert result["agent_response"] == "Improved answer"
    assert result["verified"] is True
    assert result["retry_count"] == 1


def test_workflow_stops_after_two_rejections():
    responses = [
        {"messages": [SimpleNamespace(content="First answer")]},
        {"messages": [SimpleNamespace(content="Second answer")]},
    ]

    rejection = SimpleNamespace(
        approved=False,
        issues=["Response needs improvement"],
        reason="Insufficient response",
    )

    with (
        patch(
            "multi_agent_assistant.graph.workflows."
            "technical_agent.invoke",
            side_effect=responses,
        ) as mock_agent,
        patch(
            "multi_agent_assistant.graph.workflows.verify_response",
            return_value=rejection,
        ) as mock_verifier,
    ):
        result = workflow.invoke(initial_state())

    assert mock_agent.call_count == 2
    assert mock_verifier.call_count == 2
    assert result["verified"] is False
    assert result["retry_count"] == 2
    assert result["agent_response"] == "Second answer"
