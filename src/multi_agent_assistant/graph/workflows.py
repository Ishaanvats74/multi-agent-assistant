from langgraph.graph import StateGraph, START, END

from .state import AgentState
from ..agents.verifier_agent import verify_response
from ..agents.technical_agent import technical_agent
from ..agents.planning_agent import planning_agent
from ..agents.finance_agent import finance_agent


def technical_node(state: AgentState):

    result = technical_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": state["user_message"]
            }
        ]
    })

    response = result["messages"][-1].content

    return {
        "agent_response": response
    }


def planning_node(state: AgentState):

    result = planning_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": state["user_message"]
            }
        ]
    })

    response = result["messages"][-1].content

    return {
        "agent_response": response
    }



def finance_node(state: AgentState):

    result = finance_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": state["user_message"]
            }
        ]
    })

    response = result["messages"][-1].content

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

    retry_count = state.get("retry_count", 0)

    if not result.approved:
        retry_count += 1

    print("\n========== VERIFICATION ==========")
    print("Approved:", result.approved)
    print("Issues:", result.issues)
    print("Reason:", result.reason)
    print("==================================\n")

    return {
        "verified": result.approved,
        "verification_issues": result.issues,
        "verification_reason": result.reason,
        "retry_count": retry_count
    }


def verification_router(state: AgentState):

    if state["verified"]:
        return "approved"

    if state["retry_count"] >= 2:
        return "failed"

    return state["route"]


graph = StateGraph(AgentState)

# Agents
graph.add_node("technical", technical_node)
graph.add_node("planning", planning_node)
graph.add_node("finance", finance_node)

# Verification
graph.add_node("verification", verification_node)


# -------------------------
# Initial routing
# -------------------------

graph.add_conditional_edges(
    START,
    route_agent,
    {
        "technical": "technical",
        "planning": "planning",
        "finance": "finance"
    }
)


# -------------------------
# Agent → Verification
# -------------------------

graph.add_edge("technical", "verification")
graph.add_edge("planning", "verification")
graph.add_edge("finance", "verification")


# -------------------------
# Verification → Retry / END
# -------------------------

graph.add_conditional_edges(
    "verification",
    verification_router,
    {
        "approved": END,
        "failed": END,

        # Retry the SAME specialist
        "technical": "technical",
        "planning": "planning",
        "finance": "finance",
    }
)


workflow = graph.compile()