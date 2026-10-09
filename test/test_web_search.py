
import json
from unittest.mock import MagicMock, patch

from multi_agent_assistant.tools.web_searching import web_search


def test_empty_search_results():
    with (
        patch(
            "multi_agent_assistant.tools.web_searching.get_diffbot"
        ),
        patch(
            "multi_agent_assistant.tools.web_searching."
            "DiffbotWebSearchRetriever"
        ) as mock_retriever,
    ):
        mock_retriever.return_value.invoke.return_value = []

        result = web_search.func("Python documentation")

    payload = json.loads(result)

    assert payload["status"] == "no_results"
    assert payload["results"] == []


def test_search_preserves_url_and_content():
    doc = MagicMock()
    doc.metadata = {
        "title": "Official Python Documentation",
        "pageUrl": "https://docs.example.test/python",
    }
    doc.page_content = "Documentation content."

    with (
        patch(
            "multi_agent_assistant.tools.web_searching.get_diffbot"
        ),
        patch(
            "multi_agent_assistant.tools.web_searching."
            "DiffbotWebSearchRetriever"
        ) as mock_retriever,
    ):
        mock_retriever.return_value.invoke.return_value = [doc]

        result = web_search.func("Python documentation")

    payload = json.loads(result)

    assert payload["status"] == "success"
    assert payload["results"][0]["url"] == (
        "https://docs.example.test/python"
    )
    assert payload["results"][0]["content"] == (
        "Documentation content."
    )


def test_search_timeout_returns_safe_error():
    with (
        patch(
            "multi_agent_assistant.tools.web_searching.get_diffbot"
        ),
        patch(
            "multi_agent_assistant.tools.web_searching."
            "DiffbotWebSearchRetriever"
        ) as mock_retriever,
    ):
        mock_retriever.return_value.invoke.side_effect = TimeoutError(
            "private provider details"
        )

        result = web_search.func("Python documentation")

    assert "timed out" in result.lower() or "failed" in result.lower()
    assert "private provider details" not in result


def test_search_handles_provider_failure():
    with (
        patch(
            "multi_agent_assistant.tools.web_searching.get_diffbot"
        ),
        patch(
            "multi_agent_assistant.tools.web_searching."
            "DiffbotWebSearchRetriever"
        ) as mock_retriever,
    ):
        mock_retriever.return_value.invoke.side_effect = Exception(
            "private provider details"
        )

        result = web_search.func("Python documentation")

    assert "private provider details" not in result
    assert "failed" in result.lower() or "unavailable" in result.lower()


def test_search_rejects_empty_query():
    result = web_search.func("   ")

    assert "empty" in result.lower()
