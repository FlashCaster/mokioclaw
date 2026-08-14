"""M8a 测试：意图路由（路由函数 + confidence 解析，不依赖 LLM）。"""

from mokioclaw.graph.nodes import _coerce_confidence, intent_route_fn
from mokioclaw.graph.workflow import build_entry_workflow


# ---- 路由函数 ----

def test_intent_route_fn_chat():
    assert intent_route_fn({"intent_route": "chat"}) == "chat_responder"


def test_intent_route_fn_workflow():
    assert intent_route_fn({"intent_route": "workflow"}) == "planner"


def test_intent_route_fn_default_to_workflow():
    # 未设置 intent_route 时默认走 workflow（安全兜底）
    assert intent_route_fn({}) == "planner"


# ---- confidence 解析 ----

def test_coerce_confidence_valid():
    assert _coerce_confidence(0.8) == 0.8
    assert _coerce_confidence("0.9") == 0.9


def test_coerce_confidence_invalid():
    assert _coerce_confidence("not-a-number") == 0.0
    assert _coerce_confidence(None) == 0.0


def test_coerce_confidence_clamp():
    assert _coerce_confidence(1.5) == 1.0
    assert _coerce_confidence(-0.5) == 0.0


# ---- 入口图结构 ----

def test_entry_workflow_compiles():
    graph = build_entry_workflow()
    assert graph is not None
