"""M5 单元测试：三层记忆 + Notepad + 上下文压缩路由（不依赖 LLM）。"""

from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    persist_history_summary,
    read_history_summary,
)
from mokioclaw.graph.nodes import context_compressor_route, context_monitor_route
from mokioclaw.tools.notepad_tool import append_notepad, read_notepad


# ---- Notepad ----

def test_append_and_read_notepad(tmp_path):
    state = RuntimeState(workspace=tmp_path)
    result = append_notepad(state, "关键发现", "测试通过")
    assert result["ok"] is True
    read = read_notepad(state)
    assert read["exists"] is True
    assert "关键发现" in read["content"]
    assert "测试通过" in read["content"]


def test_append_empty_content_rejected(tmp_path):
    state = RuntimeState(workspace=tmp_path)
    result = append_notepad(state, "空", "   ")
    assert result["ok"] is False
    assert "empty" in result["error"]


def test_read_notepad_missing(tmp_path):
    state = RuntimeState(workspace=tmp_path)
    read = read_notepad(state)
    assert read["exists"] is False
    assert read["content"] == ""


# ---- 三层记忆 ----

def test_build_layered_memory_three_layers(tmp_path):
    state = {
        "runtime": RuntimeState(workspace=tmp_path),
        "task": "test task",
        "todos": [{"id": "todo-1", "content": "x", "status": "pending", "note": ""}],
        "plan_summary": "plan",
        "acceptance_criteria": ["c"],
        "verification_commands": ["cmd"],
    }
    memory = build_layered_memory(state, node="planner")
    assert set(memory.keys()) == {"rules", "working_memory", "history_summary_store"}
    assert memory["working_memory"]["task"] == "test task"
    assert memory["working_memory"]["todos"][0]["id"] == "todo-1"
    assert memory["rules"]["scope"] == "workspace"
    assert memory["history_summary_store"]["history_exists"] is False
    assert memory["history_summary_store"]["notepad_exists"] is False


def test_format_layered_memory_for_prompt(tmp_path):
    state = {"runtime": RuntimeState(workspace=tmp_path), "task": "t"}
    memory = build_layered_memory(state, node="planner")
    text = format_layered_memory_for_prompt(memory)
    assert "working_memory" in text
    assert "rules" in text


# ---- 历史摘要 ----

def test_persist_and_read_history_summary(tmp_path):
    state = RuntimeState(workspace=tmp_path)
    result = persist_history_summary(state, "压缩后的摘要")
    assert result["ok"] is True
    read = read_history_summary(state)
    assert read["exists"] is True
    assert "压缩后的摘要" in read["content"]


# ---- 上下文压缩路由 ----

def test_context_monitor_route_compress():
    assert context_monitor_route({"context_should_compress": True}) == "context_compressor"


def test_context_monitor_route_normal():
    state = {"context_should_compress": False, "context_next_node": "verifier"}
    assert context_monitor_route(state) == "verifier"


def test_context_monitor_route_default():
    # 未设 next_node 时默认去 verifier
    assert context_monitor_route({"context_should_compress": False}) == "verifier"


def test_context_compressor_route():
    assert context_compressor_route({"context_next_node": "planner"}) == "planner"
