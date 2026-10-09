
import pytest

from multi_agent_assistant.graph.workflows import (
    route_agent,
    verification_router,
)


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("technical", "technical"),
        ("planning", "planning"),
        ("financial", "financial"),
    ],
)
def test_valid_routes(route, expected):
    assert route_agent({"route": route}) == expected


@pytest.mark.parametrize(
    "route",
    ["", "unknown", None, "finance", "TECHNICAL"],
)
def test_invalid_routes_are_rejected(route):
    with pytest.raises((ValueError, TypeError)):
        route_agent({"route": route})


def test_missing_route_is_rejected():
    with pytest.raises((ValueError, TypeError, KeyError)):
        route_agent({})


def test_verification_approval_ends_workflow():
    state = {
        "verified": True,
        "retry_count": 0,
        "route": "technical",
    }

    assert verification_router(state) == "approved"


def test_verification_rejection_allows_retry():
    state = {
        "verified": False,
        "retry_count": 1,
        "route": "planning",
    }

    assert verification_router(state) == "planning"


def test_verification_rejection_stops_at_limit():
    state = {
        "verified": False,
        "retry_count": 2,
        "route": "technical",
    }

    assert verification_router(state) == "failed"
