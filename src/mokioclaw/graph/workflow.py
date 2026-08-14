"""Workflow assembly for the planner→actor→verifier graph (M4)."""

from langgraph.graph import END, START, StateGraph

from mokioclaw.graph.nodes import (
    actor_node,
    final_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from mokioclaw.graph.state import MokioGraphState


def build_workflow():
    """Build the M4 LangGraph workflow.

    Graph shape:
        START → planner → actor → verifier ─passed/耗尽→ final → END
                              ↑            │
                              └──failed 且 attempts<max──┘
    """
    graph = StateGraph(MokioGraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {"planner": "planner", "final": "final"},
    )
    graph.add_edge("final", END)

    return graph.compile()
