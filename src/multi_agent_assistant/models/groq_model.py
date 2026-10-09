import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)


def get_groq() -> ChatGroq:
    token = os.getenv("GROQ_API_KEY", "").strip()

    if not token:
        raise RuntimeError("Missing required configuration: GROQ_API_KEY.")

    try:
        return ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.5,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=30,
            max_retries=2,
            api_key=token,
        )
    
    except Exception:
        logger.exception("Failed to initialize the Groq client")
        raise RuntimeError("The language model client could not be initialized.") from None

llm = get_groq()
