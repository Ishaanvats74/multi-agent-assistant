from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from ..tools.web_searching import web_search

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=API_KEY
)

llm_with_tools = llm.bind_tools([web_search])
