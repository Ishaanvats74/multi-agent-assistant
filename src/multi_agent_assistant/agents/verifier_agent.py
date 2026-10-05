from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from ..models.groq_model import llm


class VerificationResult(BaseModel):
    approved: bool = Field(description="True if the response is good enough to return.")

    issues: List[str] = Field(description="Specific issues found in the response.")

    reason: str = Field(description="Short explanation of the verification decision.")


verifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
            """
                You are a verification agent in a multi-agent system.

                Evaluate the generated response against the original user request.

                Check:

                - relevance
                - completeness
                - correctness
                - user constraints
                - clarity
                - whether the response actually solves the request

                Approve the response only if it is good enough to be returned
                to the user.

                Do not rewrite the response.
                Do not solve the task yourself.
                Only evaluate it.
            """
        ),
        (
        "human",
            """
                USER REQUEST:
                {user_message}

                GENERATED RESPONSE:
                {agent_response}
            """
        )
])


verifier = verifier_prompt | llm.with_structured_output(VerificationResult)


def verify_response(user_message: str, agent_response: str) -> VerificationResult:
    
    return verifier.invoke({
        "user_message": user_message,
        "agent_response": agent_response
    })