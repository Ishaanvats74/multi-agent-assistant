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

    prompt = state["user_message"]

    # If this is a retry, provide the previous response
    # and verification feedback to the planning agent.
    if state.get("verification_issues"):
        prompt = f"""
Original user request:
{state["user_message"]}

Previous response:
{state["agent_response"]}

The verification agent rejected the previous response
for the following reasons:

{state["verification_issues"]}

Verification explanation:
{state.get("verification_reason", "")}

Revise the previous response and fix all of the issues identified
by the verification agent.

Return only the improved response.
"""

    result = planning_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
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

    if route == "financial":
        return "financial"

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
graph.add_node("financial", finance_node)

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
        "financial": "financial"
    }
)


# -------------------------
# Agent → Verification
# -------------------------

graph.add_edge("technical", "verification")
graph.add_edge("planning", "verification")
graph.add_edge("financial", "verification")


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
        "financial": "financial",
    }
)


workflow = graph.compile()