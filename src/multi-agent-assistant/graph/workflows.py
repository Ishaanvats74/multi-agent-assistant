from langgraph.graph import StateGraph, START, END

from .state import AgentState
from ..agents.verifier_agent import verify_response
from ..agents.technical_agent import technical_agent
from ..agents.planning_agent import planning_agent
from ..agents.finance_agent import finance_agent


def technical_node(state: AgentState):
    response = technical_agent(state["user_message"])

    return {
        "agent_response": response
    }


def planning_node(state: AgentState):
    response = planning_agent(state["user_message"])

    return {
        "agent_response": response
    }


def finance_node(state: AgentState):
    response = finance_agent(state["user_message"])

    return {
        "agent_response": response
    }


def route_agent(state: AgentState):

    route = state["route"]

    if route == "technical":
        return "technical"

    if route == "planning":
        return "planning"

    if route == "finance":
        return "finance"

    raise ValueError(f"Unknown route: {route}")

def verification_node(state: AgentState):

    result = verify_response(
        state["user_message"],
        state["agent_response"]
    )

    print("\n========== VERIFICATION ==========")
    print("Approved:", result.approved)
    print("Issues:", result.issues)
    print("Reason:", result.reason)
    print("==================================\n")

    return {
        "verified": result.approved,
        "verification_issues": result.issues,
        "verification_reason": result.reason
    }
def verification_router(state: AgentState):

    if state["verified"]:
        return "approved"

    if state["retry_count"] >= 2:
        return "failed"

    return "retry"


def retry_router(state: AgentState):

    if state["route"] == "technical":
        return "technical"

    if state["route"] == "planning":
        return "planning"

    if state["route"] == "finance":
        return "finance"

    raise ValueError(f"Unknown route: {state['route']}")


graph = StateGraph(AgentState)

graph.add_node("technical", technical_node)
graph.add_node("planning", planning_node)
graph.add_node("finance", finance_node)
graph.add_node("verification", verification_node)

graph.add_conditional_edges(
    START,
    route_agent,
    {
        "technical": "technical",
        "planning": "planning",
        "finance": "finance"
    }
)

graph.add_conditional_edges(
    "verification",
    verification_router,
    {
        "approved": END,
        "failed": END,
        "technical": "technical",
        "planning": "planning",
        "finance": "finance",
    }
)
graph.add_edge("technical", "verification")
graph.add_edge("planning", "verification")
graph.add_edge("finance", "verification")

workflow = graph.compile()