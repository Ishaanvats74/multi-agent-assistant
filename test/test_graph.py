import pytest

from multi_agent_assistant.graph.state import AgentState
from multi_agent_assistant.graph.workflows import route_agent,verification_router


def make_state(**overrides):
    state: AgentState = {
        "user_message": "Test message",
        "route": "planning",
        "agent_response": "Test response",
        "verified": False,
        "verification_issues": [],
        "verification_reason": "",
        "retry_count": 0,
    }

    state.update(overrides)

    return state


@pytest.mark.parametrize(
    "route",
    [
        "technical",
        "planning",
        "financial",
    ],
)
def test_route_agent(route):
    state = make_state(route=route)

    assert route_agent(state) == route


def test_unknown_route():
    state = make_state(route="unknown")

    with pytest.raises(ValueError):
        route_agent(state)


def test_verification_router_approved():
    state = make_state(verified=True,retry_count=0)

    assert verification_router(state) == "approved"


def test_verification_router_retry():
    state = make_state(verified=False,retry_count=1,route="planning")

    assert verification_router(state) == "planning"


def test_verification_router_max_retries():
    state = make_state(verified=False,retry_count=2,route="planning")

    assert verification_router(state) == "failed"
