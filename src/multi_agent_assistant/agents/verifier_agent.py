import logging

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from ..models.groq_model import llm

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    approved: bool = Field(description="True only if the response adequately satisfies the user request.")

    issues: list[str] = Field(default_factory=list,description="Specific unresolved issues. Empty when no issues are found.")

    reason: str = Field(min_length=1,description="Short explanation of the verification decision.")


verifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
            You are the verification agent in a multi-agent assistant system.

            Evaluate the generated response against the original user request.

            Check:
            - Relevance to the request
            - Factual and technical correctness
            - Completeness
            - Compliance with user constraints
            - Clarity and usefulness
            - Whether the response actually addresses the request

            Rules:
            1. Do not solve the original task yourself.
            2. Do not rewrite or improve the response.
            3. Evaluate the response, not the writing style alone.
            4. Do not reject a response merely because it uses a different
            format from your preferred format.
            5. Identify specific, actionable issues.
            6. Do not invent errors or demand unsupported details.
            7. Approve only when the response is good enough to return.
            8. If the response is empty or fails to address the request,
            reject it and explain why.
            9. If approved, return an empty issues list.
            10. If rejected, include the important issues that must be fixed.
            11. Keep the reason concise and consistent with the decision.
        """
        ),(
        "human",
        """
            USER REQUEST:
            {query_input}

            GENERATED RESPONSE:
            {agent_response}
        """
    )
])

verifier = verifier_prompt | llm.with_structured_output(VerificationResult)


def verify_response(query_input: str,agent_response: str) -> VerificationResult:

    if not isinstance(query_input, str) or not query_input.strip():
        raise ValueError("query_input must be a non-empty string.")

    if not isinstance(agent_response, str) or not agent_response.strip():
        return VerificationResult(approved=False,issues=["The generated response is empty."],reason="The agent did not generate a usable response.")

    try:
        result = verifier.invoke({
            "query_input": query_input.strip(),
            "agent_response": agent_response.strip()
        })

    except Exception:
        logger.exception("Response verification failed")
        raise RuntimeError("The verification service is temporarily unavailable.") from None

    return result
