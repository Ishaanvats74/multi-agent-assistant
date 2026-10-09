from typing import TypedDict


class AgentState(TypedDict):
    query_input: str
    route: str
    agent_response: str

    verified: bool
    verification_issues: list[str]
    verification_reason: str

    retry_count: int
