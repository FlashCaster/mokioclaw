"""todo_tool 单元测试：规划器核心（规范化/更新/渲染/持久化）。"""

from mokioclaw.core.state import RuntimeState
from mokioclaw.tools.todo_tool import (
    persist_todos,
    render_todo_markdown,
    todo_items_from_strings,
    update_todo,
    write_todos,
)


def test_write_todos_normalizes_list():
    result = write_todos(
        ["写测试", "写实现"],
        ["测试通过"],
        ["python -m pytest -q"],
    )
    assert result["ok"] is True
    assert result["todos"] == ["写测试", "写实现"]


def test_write_todos_normalizes_json_string():
    # LLM 有时返回 JSON 字符串而非数组，需兼容
    result = write_todos('["a", "b"]', '["c"]', '["d"]')
    assert result["ok"] is True
    assert result["todos"] == ["a", "b"]


def test_write_todos_markdown_lines():
    # markdown 列表逐行拆
    result = write_todos("- 任务一\n- 任务二", ["验收"], ["命令"])
    assert result["todos"] == ["任务一", "任务二"]


def test_write_todos_requires_all_three():
    # 三要素缺一不可，否则 ok=False
    result = write_todos(["a"], [], ["c"])
    assert result["ok"] is False


def test_todo_items_generate_ids():
    items = todo_items_from_strings(["写测试", "写实现"])
    assert items[0]["id"] == "todo-1"
    assert items[0]["status"] == "pending"
    assert items[1]["id"] == "todo-2"


def test_todo_items_preserve_existing_status():
    # 计划修订时保留已有进度
    existing = [{"id": "todo-1", "content": "写测试", "status": "completed", "note": "done"}]
    items = todo_items_from_strings(["写测试", "写实现"], existing=existing)
    assert items[0]["status"] == "completed"
    assert items[0]["note"] == "done"
    assert items[1]["status"] == "pending"


def test_update_todo_status():
    items = todo_items_from_strings(["写测试"])
    result = update_todo(items, "todo-1", "in_progress")
    assert result["ok"] is True
    assert result["todos"][0]["status"] == "in_progress"


def test_update_todo_invalid_status():
    items = todo_items_from_strings(["写测试"])
    result = update_todo(items, "todo-1", "flying")
    assert result["ok"] is False
    assert "status must be" in result["error"]


def test_update_todo_unknown_id():
    items = todo_items_from_strings(["写测试"])
    result = update_todo(items, "todo-99", "completed")
    assert result["ok"] is False
    assert "unknown todo_id" in result["error"]


def test_render_todo_markdown():
    items = todo_items_from_strings(["写测试", "写实现"])
    md = render_todo_markdown(items, ["测试通过"], ["python -m pytest -q"], "做一个计算器")
    assert "# MokioClaw Todo" in md
    assert "## Plan" in md
    assert "做一个计算器" in md
    assert "todo-1" in md
    assert "## Acceptance Criteria" in md
    assert "## Verification Commands" in md


def test_persist_todos_writes_file(tmp_path):
    state = RuntimeState(workspace=tmp_path)
    items = todo_items_from_strings(["写测试"])
    result = persist_todos(state, items, ["测试通过"], ["pytest"], "计划摘要")
    assert result["ok"] is True
    todo_file = tmp_path / "TODO.md"
    assert todo_file.exists()
    content = todo_file.read_text(encoding="utf-8")
    assert "计划摘要" in content
    assert "todo-1" in content
