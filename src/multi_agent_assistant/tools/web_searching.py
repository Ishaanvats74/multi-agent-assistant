import os
from dotenv import load_dotenv
from diffbot import Diffbot
from langchain_core.tools import tool
from langchain_diffbot import DiffbotWebSearchRetriever

load_dotenv()

db = Diffbot(token=os.environ["DIFFBOT_API_TOKEN"])

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

    docs = retriever.invoke(query)

    results = []

    for doc in docs:
        results.append(
            f"Title: {doc.metadata.get('title', '')}\n"
            f"URL: {doc.metadata.get('pageUrl', '')}\n"
            f"Content: {doc.page_content[:1500]}"
        )

    return "\n\n---\n\n".join(results)