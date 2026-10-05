from typing import TypedDict, List


class AgentState(TypedDict):
    user_message: str
    route: str
    agent_response: str

    verified: bool
    verification_issues: List[str]
    verification_reason: str

    retry_count: int