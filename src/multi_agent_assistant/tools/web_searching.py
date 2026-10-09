import os
import json
import logging

from diffbot import Diffbot
from langchain_core.tools import tool
from langchain_diffbot import DiffbotWebSearchRetriever

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
MAX_CONTENT_LENGTH = 1500


def get_diffbot():
    token = os.getenv("DIFFBOT_API_TOKEN", "").strip()

    if not token:
        raise RuntimeError("DIFFBOT_API_TOKEN is required to use web search.")

    return Diffbot(token=token)


@tool
def web_search(query: str) -> str:
    """Search the web for current information and return relevant results."""

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "status": "error",
            "message": "The search query is empty.",
            "results": [],
        })

    try:
        client = get_diffbot()
        retriever = DiffbotWebSearchRetriever(client=client,k=MAX_RESULTS,fields=["title", "pageUrl", "score"])
        docs = retriever.invoke(query.strip())

        results = []
        for doc in docs:
            metadata = doc.metadata or {}
            content = (doc.page_content or "").strip()

            results.append({
                "title": metadata.get("title") or "Untitled",
                "url": metadata.get("pageUrl") or "",
                "content": (
                    content or "No page content was returned."
                )[:MAX_CONTENT_LENGTH],
            })

        return json.dumps({
            "status": "success" if results else "no_results",
            "results": results,
        }, ensure_ascii=False)

    except Exception:
        logger.exception("Diffbot web search failed")
        return json.dumps({
            "status": "error",
            "message": "Web search failed or timed out.",
            "results": [],
        })
