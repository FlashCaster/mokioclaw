"""Workflow assembly for the planner→verifier graph (M6 multi-agent).

Graph shape (actor replaced by codeAgent delegation inside planner):
    START → planner(supervisor，委派 searchAgent/codeAgent)
             → context_monitor ─超限→ context_compressor
                    │正常              │
                    ↓                 ↓
                 verifier ←───────────┘
                    │
                    ↓
              context_monitor → planner(重试) / final
"""

from langgraph.graph import END, START, StateGraph

from mokioclaw.graph.nodes import (
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    final_node,
    planner_node,
    verifier_node,
)
from mokioclaw.graph.state import MokioGraphState


def build_workflow():
    graph = StateGraph(MokioGraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("context_monitor", context_monitor_node)
    graph.add_node("context_compressor", context_compressor_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")
    graph.add_conditional_edges(
        "context_monitor",
        context_monitor_route,
        {
            "context_compressor": "context_compressor",
            "verifier": "verifier",
            "planner": "planner",
            "final": "final",
        },
    )
    graph.add_conditional_edges(
        "context_compressor",
        context_compressor_route,
        {"verifier": "verifier", "planner": "planner", "final": "final"},
    )
    graph.add_edge("verifier", "context_monitor")
    graph.add_edge("final", END)

    return graph.compile()
