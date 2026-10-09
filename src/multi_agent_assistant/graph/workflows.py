import logging

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from ..agents.verifier_agent import verify_response
from ..agents.technical_agent import technical_agent
from ..agents.planning_agent import planning_agent
from ..agents.finance_agent import finance_agent

logger = logging.getLogger(__name__)

ALLOWED_ROUTES = {"technical", "planning", "financial"}
MAX_REJECTIONS = 2


def build_agent_prompt(state: AgentState) -> str:
    query_input = state.get("query_input", "").strip()

    if not query_input:
        raise ValueError("query_input cannot be empty.")

    issues = state.get("verification_issues") or []

    if not issues:
        return query_input

    return f"""
            Original user request:
            {query_input}

            Previous response:
            {state.get("agent_response", "")}

            Verification issues:
            {issues}

            Verification explanation:
            {state.get("verification_reason", "")}

            Revise the previous response to address every verification issue.
            Preserve correct information and do not introduce unsupported claims.
            Return only the improved response.
            """.strip()


def extract_response(result, agent_name: str) -> str:
    if not isinstance(result, dict):
        raise RuntimeError(f"{agent_name} returned an invalid result.")

    messages = result.get("messages")

    if not isinstance(messages, (list, tuple)) or not messages:
        raise RuntimeError(f"{agent_name} returned no messages.")

    content = getattr(messages[-1], "content", None)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{agent_name} returned an empty response.")

    return content.strip()


def run_specialist(agent, state: AgentState, agent_name: str):
    prompt = build_agent_prompt(state)

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    response = extract_response(result, agent_name)

    logger.info("%s generated a response", agent_name)

    return {"agent_response": response}


# -------------------------
# Specialist nodes
# -------------------------

def technical_node(state: AgentState):
    return run_specialist(technical_agent, state, "Technical agent")


def planning_node(state: AgentState):
    return run_specialist(planning_agent, state, "Planning agent")


def finance_node(state: AgentState):
    return run_specialist(finance_agent, state, "Finance agent")


# -------------------------
# Initial routing
# -------------------------

def route_agent(state: AgentState):
    route = state.get("route")

    if route not in ALLOWED_ROUTES:
        raise ValueError("Invalid agent route.")

    return route


# -------------------------
# Verification
# -------------------------

def verification_node(state: AgentState):
    query_input = state.get("query_input", "")
    response = state.get("agent_response", "")

    if not isinstance(response, str) or not response.strip():
        approved = False
        issues = ["The agent returned an empty response."]
        reason = "A usable response was not generated."

    else:
        result = verify_response(query_input, response)

        approved = result.approved
        issues = result.issues
        reason = result.reason

    retry_count = state.get("retry_count", 0)

    if (isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0):
        raise ValueError("Invalid retry_count in workflow state.")

    if not approved:
        retry_count += 1

    return {
        "verified": approved,
        "verification_issues": issues,
        "verification_reason": reason,
        "retry_count": retry_count,
    }


def verification_router(state: AgentState):
    if state.get("verified") is True:
        return "approved"

    if state.get("retry_count", 0) >= MAX_REJECTIONS:
        return "failed"

    return route_agent(state)



# -------------------------
# Build the graph
# -------------------------

graph = StateGraph(AgentState)

graph.add_node("technical", technical_node)
graph.add_node("planning", planning_node)
graph.add_node("financial", finance_node)
graph.add_node("verification", verification_node)

graph.add_conditional_edges(
    START,
    route_agent,
    {
        "technical": "technical",
        "planning": "planning",
        "financial": "financial"
    }
)

graph.add_edge("technical", "verification")
graph.add_edge("planning", "verification")
graph.add_edge("financial", "verification")

graph.add_conditional_edges(
    "verification",
    verification_router,
    {
        "approved": END,
        "failed": END,
        "technical": "technical",
        "planning": "planning",
        "financial": "financial"
    }
)

workflow = graph.compile()
