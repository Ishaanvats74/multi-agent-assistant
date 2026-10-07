import os
from dotenv import load_dotenv
from diffbot import Diffbot
from langchain_core.tools import tool
from langchain_diffbot import DiffbotWebSearchRetriever

load_dotenv()

def get_diffbot():
    token = os.getenv("DIFFBOT_API_TOKEN")

    if not token:
        raise RuntimeError(
            "DIFFBOT_API_TOKEN is required to use web search."
        )

    return Diffbot(token=token)



retriever = DiffbotWebSearchRetriever(
    client=db,
    k=5,
    fields=["title", "pageUrl", "score"],
)


@tool
def web_search(query: str) -> str:
    """Search the web for current information and return relevant results."""
    print("\n🔎 WEB SEARCH CALLED")
    print("Query:", query)
    db = get_diffbot()

    docs = retriever.invoke(query)

    results = []

    for doc in docs:
        results.append(
            f"Title: {doc.metadata.get('title', '')}\n"
            f"URL: {doc.metadata.get('pageUrl', '')}\n"
            f"Content: {doc.page_content[:1500]}"
        )

    return "\n\n---\n\n".join(results)