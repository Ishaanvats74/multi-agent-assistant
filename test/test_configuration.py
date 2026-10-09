
import pytest

from multi_agent_assistant.models.groq_model import get_groq
from multi_agent_assistant.tools.web_searching import get_diffbot


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Missing required configuration: GROQ_API_KEY."):
        get_groq()


def test_diffbot_requires_api_token(monkeypatch):
    monkeypatch.delenv("DIFFBOT_API_TOKEN", raising=False)

    with pytest.raises(
        RuntimeError,
        match="DIFFBOT_API_TOKEN is required to use web search.",
    ):
        get_diffbot()


def test_groq_rejects_whitespace_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    with pytest.raises(RuntimeError, match="Missing required configuration: GROQ_API_KEY."):
        get_groq()


def test_diffbot_rejects_whitespace_token(monkeypatch):
    monkeypatch.setenv("DIFFBOT_API_TOKEN", "   ")

    with pytest.raises(
        RuntimeError,
        match="DIFFBOT_API_TOKEN is required to use web search.",
    ):
        get_diffbot()
