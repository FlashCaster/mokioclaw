"""M7 第一阶段测试：trace 链路观测（不依赖 LLM）。"""

import json

from mokioclaw.core.trace import TraceWriter


def test_record_writes_events(tmp_path):
    writer = TraceWriter(tmp_path)
    writer.record("run_start", task="t")
    writer.record_tool_call("planner", "TodoWriteTool", {"todos": ["a"]})
    writer.record_tool_result("planner", "TodoWriteTool", True)
    writer.record_handoff("planner", "codeAgent", "实现")
    summary = writer.finalize(
        {"passed": True, "todos": [], "agent_handoffs": [], "sources": []}
    )

    assert summary["event_count"] == 4
    assert summary["counters"]["tool_calls"] == 1
    assert summary["counters"]["tool_results"] == 1
    assert summary["counters"]["handoffs"] == 1

    events_path = tmp_path / ".mokioclaw" / "trace" / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    for line in lines:
        event = json.loads(line)
        assert "ts" in event and "type" in event


def test_finalize_writes_summary_and_timeline(tmp_path):
    writer = TraceWriter(tmp_path)
    writer.record("node_start", node="planner")
    writer.record("node_end", node="planner")
    writer.finalize(
        {
            "passed": True,
            "attempts": 1,
            "todos": [],
            "agent_handoffs": [],
            "sources": [],
        }
    )

    trace_dir = tmp_path / ".mokioclaw" / "trace"
    assert (trace_dir / "summary.json").exists()
    assert (trace_dir / "timeline.md").exists()
    summary = json.loads((trace_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final"]["passed"] is True
    assert summary["final"]["attempts"] == 1


def test_node_counter_accumulates(tmp_path):
    writer = TraceWriter(tmp_path)
    writer.record("node_start", node="planner")
    writer.record("node_end", node="planner")
    writer.record("node_start", node="planner")
    writer.record("node_end", node="planner")
    summary = writer.finalize(None)
    # 每个 node 事件计数（start+end 各 2 次 = 4）
    assert summary["counters"]["nodes"]["planner"] == 4


def test_timeline_mentions_events(tmp_path):
    writer = TraceWriter(tmp_path)
    writer.record_tool_call("actor", "FileWriteTool", {"path": "a.py"})
    writer.record_tool_result("actor", "FileWriteTool", True, "create")
    writer.finalize(None)
    timeline = (tmp_path / ".mokioclaw" / "trace" / "timeline.md").read_text(encoding="utf-8")
    assert "FileWriteTool" in timeline
