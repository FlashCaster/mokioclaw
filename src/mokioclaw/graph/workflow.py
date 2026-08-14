"""Workflow assembly (M8a Stage 6: intent router + complex workflow).

两个图：
  - build_entry_workflow：START → intent_router → (chat_responder / planner 出口)
    聊天类输入走 chat_responder 轻量应答；任务类输入路由到复杂工作流。
  - build_complex_workflow：planner → verifier 闭环（原 build_workflow 逻辑）。
"""

from langgraph.graph import END, START, StateGraph

from mokioclaw.graph.nodes import (
    chat_responder_node,
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    final_node,
    intent_route_fn,
    intent_router_node,
    planner_node,
    verifier_node,
)
from mokioclaw.graph.state import MokioGraphState


def build_complex_workflow():
    """任务类主流程：planner → (context_monitor → verifier) 闭环。"""
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


def build_entry_workflow():
    """入口图：意图路由。chat → chat_responder；workflow → 结束（主程序再跑 complex 图）。"""
    graph = StateGraph(MokioGraphState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("chat_responder", chat_responder_node)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        intent_route_fn,
        {"chat_responder": "chat_responder", "planner": END},
    )
    graph.add_edge("chat_responder", END)

    return graph.compile()


def build_workflow():
    """向后兼容：默认返回复杂工作流（任务类主流程）。"""
    return build_complex_workflow()
