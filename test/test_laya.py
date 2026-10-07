import pytest
from multi_agent_assistant.router.route import laya_router

@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "Create a 30-day DSA study plan",
            "planning",
        ),
        (
            "Make me a weekly schedule for learning Python",
            "planning",
        ),
        (
            "My FastAPI application is returning a 500 error",
            "technical",
        ),
        (
            "Explain how Docker networking works",
            "technical",
        ),
        (
            "Help me create a monthly budget",
            "financial",
        ),
        (
            "How much should I save every month?",
            "financial",
        ),
    ],
)

def test_intent_classification(message, expected):
    result = laya_router(message)

    assert result['intent'] == expected
