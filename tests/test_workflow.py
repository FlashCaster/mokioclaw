"""M4 工作流单元测试：路由逻辑 + JSON 解析 + 图组装（不依赖 LLM）。"""

from mokioclaw.graph.nodes import _extract_json, _normalize_checks, verifier_route
from mokioclaw.graph.workflow import build_workflow


def test_verifier_route_passed():
    assert verifier_route({"passed": True}) == "final"


def test_verifier_route_retry():
    # 未通过且未达上限 → 回 planner 重试
    state = {"passed": False, "attempts": 1, "max_attempts": 3}
    assert verifier_route(state) == "planner"


def test_verifier_route_max_attempts():
    # 未通过但已达上限 → 结束，不再重试
    state = {"passed": False, "attempts": 3, "max_attempts": 3}
    assert verifier_route(state) == "final"


def test_extract_json_plain():
    parsed = _extract_json('{"passed": true, "reason": "ok"}')
    assert parsed == {"passed": True, "reason": "ok"}


def test_extract_json_fenced():
    text = 'Here is my verdict:\n```json\n{"passed": false, "reason": "missing file"}\n```'
    parsed = _extract_json(text)
    assert parsed == {"passed": False, "reason": "missing file"}


def test_extract_json_invalid():
    assert _extract_json("no json here") is None


def test_normalize_checks():
    raw = [
        {"name": "file exists", "passed": True, "detail": "ok"},
        {"name": "output correct", "passed": False},
    ]
    checks = _normalize_checks(raw)
    assert len(checks) == 2
    assert checks[0]["passed"] is True
    assert checks[1]["detail"] == ""


def test_normalize_checks_non_list():
    assert _normalize_checks("not a list") == []


def test_build_workflow_nodes():
    graph = build_workflow()
    nodes = set(graph.get_graph().nodes)
    assert {"planner", "verifier", "final", "context_monitor", "context_compressor"} <= nodes
