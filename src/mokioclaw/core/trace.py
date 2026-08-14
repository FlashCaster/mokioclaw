"""Trace 链路观测（M7 第一阶段）。

旁路观测：记录节点、工具、handoff、检查点、压缩事件，不改变 agent 行为。
产物（<workspace>/.mokioclaw/trace/）：
  - events.jsonl：追加的原始事件流（每行一个 JSON）
  - summary.json ：运行结束汇总（节点/工具/handoff/检查点/压缩计数 + 最终状态）
  - timeline.md  ：人类可读时间线
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TraceWriter:
    """旁路观测器：事件写入 workspace/.mokioclaw/trace/，记录失败不影响主流程。"""

    def __init__(self, workspace: Path):
        self.dir = Path(workspace) / ".mokioclaw" / "trace"
        self.events_path = self.dir / "events.jsonl"
        self.summary_path = self.dir / "summary.json"
        self.timeline_path = self.dir / "timeline.md"
        self.started_at = _now_iso()
        self._events: list[dict[str, Any]] = []
        self._counters: dict[str, Any] = {
            "nodes": {},
            "tool_calls": 0,
            "tool_results": 0,
            "handoffs": 0,
            "checkpoints": 0,
            "compressions": 0,
            "errors": 0,
        }
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, event: dict[str, Any]) -> None:
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass  # 旁路观测：写失败不影响主流程

    def _update_counters(self, event_type: str, node: str) -> None:
        if event_type in ("node_start", "node_end"):
            self._counters["nodes"][node] = self._counters["nodes"].get(node, 0) + 1
        elif event_type == "tool_call":
            self._counters["tool_calls"] += 1
        elif event_type == "tool_result":
            self._counters["tool_results"] += 1
        elif event_type == "handoff":
            self._counters["handoffs"] += 1
        elif event_type == "checkpoint":
            self._counters["checkpoints"] += 1
        elif event_type == "compression":
            self._counters["compressions"] += 1
        elif event_type == "error":
            self._counters["errors"] += 1

    def record(self, event_type: str, node: str = "", **data: Any) -> dict[str, Any]:
        """记录一个事件（内存累积 + 追加 events.jsonl）。"""
        event: dict[str, Any] = {"ts": _now_iso(), "type": event_type}
        if node:
            event["node"] = node
        event.update(data)
        self._events.append(event)
        self._append_jsonl(event)
        self._update_counters(event_type, node)
        return event

    def record_tool_call(self, node: str, tool_name: str, args: dict[str, Any] | None = None) -> None:
        self.record(
            "tool_call",
            node=node,
            tool=tool_name,
            args=_short_args(args or {}),
        )

    def record_tool_result(self, node: str, tool_name: str, ok: bool, detail: str = "") -> None:
        self.record(
            "tool_result",
            node=node,
            tool=tool_name,
            ok=bool(ok),
            detail=_short_text(detail, 200),
        )

    def record_handoff(
        self, from_agent: str, to_agent: str, instruction: str = ""
    ) -> None:
        self.record(
            "handoff",
            node=from_agent,
            from_agent=from_agent,
            to_agent=to_agent,
            instruction=_short_text(instruction, 200),
        )

    def finalize(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """运行结束：写 summary.json + timeline.md，返回汇总。"""
        ended_at = _now_iso()
        summary: dict[str, Any] = {
            "started_at": self.started_at,
            "ended_at": ended_at,
            "event_count": len(self._events),
            "counters": self._counters,
        }
        if state:
            summary["final"] = {
                "passed": state.get("passed"),
                "attempts": state.get("attempts"),
                "todo_count": len(state.get("todos") or []),
                "handoff_count": len(state.get("agent_handoffs") or []),
                "source_count": len(state.get("sources") or []),
            }

        try:
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        try:
            self.timeline_path.write_text(_render_timeline(self._events), encoding="utf-8")
        except Exception:
            pass

        return summary


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------

def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _short_args(args: dict[str, Any], limit: int = 80) -> str:
    try:
        return _short_text(json.dumps(args, ensure_ascii=False, default=str), limit)
    except Exception:
        return ""


def _render_timeline(events: list[dict[str, Any]]) -> str:
    lines = ["# MokioClaw Trace 时间线", ""]
    for event in events:
        ts = event.get("ts", "")
        etype = event.get("type", "")
        node = event.get("node", "")
        if etype == "node_start":
            lines.append(f"- `{ts}` ▶ **进入节点 {node}**")
        elif etype == "node_end":
            lines.append(f"- `{ts}` ◀ 离开节点 {node}")
        elif etype == "tool_call":
            lines.append(f"- `{ts}` 🔧 调用工具 `{event.get('tool', '')}`（{event.get('args', '')}）")
        elif etype == "tool_result":
            mark = "✅" if event.get("ok") else "❌"
            lines.append(f"- `{ts}` {mark} 工具结果 `{event.get('tool', '')}`：{event.get('detail', '')}")
        elif etype == "handoff":
            lines.append(
                f"- `{ts}` 🔀 handoff：{event.get('from_agent', '')} → {event.get('to_agent', '')}"
            )
        elif etype == "checkpoint":
            lines.append(f"- `{ts}` 💾 检查点：{event.get('id', '')}（{event.get('mode', '')}）")
        elif etype == "compression":
            lines.append(f"- `{ts}` 🗜️ 上下文压缩（{event.get('detail', '')}）")
        elif etype == "error":
            lines.append(f"- `{ts}` ⚠️ 错误：{event.get('detail', '')}")
        else:
            lines.append(f"- `{ts}` [{etype}]")
    return "\n".join(lines) + "\n"
