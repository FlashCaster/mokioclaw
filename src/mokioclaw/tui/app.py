"""MokioClaw Textual TUI（M7c）。

交互式终端界面：事件时间线 + 状态面板 + Memory Snapshot 面板 + 审批弹窗 + 任务输入。
架构：agent 在后台线程同步运行（graph.stream），事件通过 EventBus 推给 TUI；
BashTool 高危命令经 ApprovalBridge 阻塞等待用户在 TUI 弹窗里批准 / 拒绝。
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from mokioclaw.core.checkpoint import save_checkpoint
from mokioclaw.core.events import ApprovalBridge, EventBus
from mokioclaw.core.state import RuntimeState
from mokioclaw.core.trace import TraceWriter
from mokioclaw.graph.memory import build_layered_memory, memory_event
from mokioclaw.graph.workflow import build_workflow


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class ApprovalModal(ModalScreen[bool]):
    """审批弹窗：高危命令等待用户批准 / 拒绝。"""

    BINDINGS = [
        ("y", "approve", "批准"),
        ("n", "deny", "拒绝"),
        ("escape", "deny", "拒绝"),
    ]

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Label("[b]⚠️ 高危命令，批准执行？[/b]", id="approval-title")
            yield Static(self.command, id="approval-command")
            yield Label("[dim]按 y 批准 / n 拒绝（默认拒绝）[/dim]")
            with Horizontal(id="approval-buttons"):
                yield Button("批准 (y)", variant="warning", id="approve")
                yield Button("拒绝 (n)", variant="error", id="deny")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self.dismiss(True)
        elif event.button.id == "deny":
            self.dismiss(False)


class MokioClawApp(App):
    """MokioClaw 交互式 TUI。"""

    TITLE = "MokioClaw"
    SUB_TITLE = "planner → verifier multi-agent"
    CSS = """
    #timeline {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }
    #status-panel, #memory-panel {
        height: 1fr;
        border: solid $secondary;
        padding: 0 1;
    }
    #left {
        width: 3fr;
    }
    #right {
        width: 1fr;
    }
    #approval-box {
        width: 70;
        height: auto;
        border: thick $warning;
        padding: 1 2;
    }
    #approval-command {
        margin: 1 0;
        color: $text-warning;
    }
    """

    def __init__(
        self,
        workspace: Path,
        approval_mode: str = "inline",
        checkpoint_mode: str = "light",
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.bus = EventBus()
        self.bridge = ApprovalBridge()
        self.trace = TraceWriter(workspace, bus=self.bus)
        self.state = RuntimeState(
            workspace=workspace,
            approval_mode=approval_mode,
            checkpoint_mode=checkpoint_mode,
            trace=self.trace,
            approval_callback=self.bridge.request,  # 高危命令走 TUI 审批桥
        )
        self._approval_in_progress = False
        self._running = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                yield RichLog(id="timeline", highlight=True, markup=False, wrap=True)
                yield Input(placeholder="输入任务，回车执行（如：帮我创建一个贪吃蛇游戏）")
            with Vertical(id="right"):
                yield Static("等待任务…", id="status-panel")
                yield Static("Memory Snapshot 暂无", id="memory-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.15, self._poll)

    # ------------------------------------------------------------------
    # 事件轮询（asyncio 侧）
    # ------------------------------------------------------------------

    async def _poll(self) -> None:
        # 1) 审批请求：弹 Modal 阻塞等待用户决定，回传 ApprovalBridge
        command = self.bridge.poll_request()
        if command and not self._approval_in_progress:
            self._approval_in_progress = True
            approved = await self.push_screen_wait(ApprovalModal(command))
            self.bridge.resolve(command, bool(approved))
            self._approval_in_progress = False
            self._write(f"[bold yellow]审批[/] {'批准' if approved else '拒绝'}：{command[:60]}")

        # 2) 事件：更新时间线 / 状态 / 记忆面板
        for event in self.bus.drain():
            self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        ts = event.get("ts", "")
        etype = event.get("type", "")
        node = event.get("node", "")

        if etype == "node_start":
            self._write(f"[cyan]▶ 进入节点[/] [bold]{node}[/]")
        elif etype == "node_end":
            self._write(f"[cyan]◀ 离开节点[/] [bold]{node}[/]")
        elif etype == "tool_call":
            self._write(f"[green]🔧 {event.get('tool', '')}[/] {event.get('args', '')[:80]}")
        elif etype == "tool_result":
            mark = "✅" if event.get("ok") else "❌"
            self._write(f"   {mark} {event.get('tool', '')} {event.get('detail', '')[:60]}")
        elif etype == "handoff":
            self._write(
                f"[magenta]🔀 handoff[/] {event.get('from_agent', '')} → {event.get('to_agent', '')}"
            )
        elif etype == "checkpoint":
            self._write(f"[blue]💾 检查点 {event.get('id', '')}[/]（{event.get('mode', '')}）")
        elif etype == "compression":
            self._write(f"[yellow]🗜️ 上下文压缩[/] {event.get('detail', '')[:60]}")
        elif etype == "error":
            self._write(f"[red]⚠️ 错误[/] {event.get('detail', '')[:120]}")
        elif etype == "run_end":
            self._write(f"[bold green]✅ 运行结束[/]")
            self._running = False
        elif etype == "status":
            self._update_status(event)
            self._update_memory(event)

    def _write(self, text: str) -> None:
        timeline = self.query_one("#timeline", RichLog)
        timeline.write(text)

    def _update_status(self, event: dict[str, Any]) -> None:
        status = self.query_one("#status-panel", Static)
        lines = [
            "[b]状态面板[/b]",
            f"节点：{event.get('node', '')}",
            f"任务：{event.get('task', '')[:40]}",
            f"Todos：{event.get('todo_done', 0)}/{event.get('todo_total', 0)} 完成",
            f"尝试：{event.get('attempts', 0)}",
            f"验证：{event.get('passed')}",
            f"Handoffs：{event.get('handoffs', 0)}",
            f"Sources：{event.get('sources', 0)}",
        ]
        status.update("\n".join(lines))

    def _update_memory(self, event: dict[str, Any]) -> None:
        memory = event.get("memory") or {}
        panel = self.query_one("#memory-panel", Static)
        working = memory.get("working_memory", {})
        history = memory.get("history_summary_store", {})
        lines = [
            "[b]Memory Snapshot[/b]",
            f"node: {event.get('node', '')}",
            f"todos: {memory.get('todo_count', len(working.get('todos', [])))}",
            f"notepad: {'有' if history.get('notepad_exists') else '无'}",
            f"history: {'有' if history.get('history_exists') else '无'}",
            f"passed: {working.get('passed')}",
            f"attempts: {working.get('attempts', 0)}",
        ]
        panel.update("\n".join(lines))

    # ------------------------------------------------------------------
    # 任务提交 + agent 后台运行
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        event.input.value = ""
        if not task:
            return
        if self._running:
            self._write("[yellow]⚠️ 已有任务在运行，请等待完成[/]")
            return
        self._running = True
        self._write(f"[bold]任务：[/]{task}")
        threading.Thread(target=self._run_agent, args=(task,), daemon=True).start()

    def _run_agent(self, task: str) -> None:
        """agent 后台线程：跑图，节点/工具事件经 TraceWriter 自动入总线。"""
        try:
            graph = build_workflow()
            initial: dict[str, Any] = {
                "task": task,
                "runtime": self.state,
                "max_attempts": 3,
                "attempts": 0,
            }
            merged: dict[str, Any] = dict(initial)
            self.trace.record("run_start", task=task, checkpoint_mode=self.state.checkpoint_mode)

            for step in graph.stream(initial):
                for node_name, data in step.items():
                    merged.update(data)
                    self.trace.record("node_start", node=node_name)
                    ckpt = save_checkpoint(self.state, merged, node=node_name)
                    if ckpt.get("ok") and not ckpt.get("skipped"):
                        self.trace.record(
                            "checkpoint",
                            node=node_name,
                            id=ckpt.get("id"),
                            mode=ckpt.get("mode"),
                        )
                    self._emit_status(merged, node_name)
                    self.trace.record("node_end", node=node_name)

            self.trace.finalize(merged)
            final = merged.get("final_answer", "")
            if final:
                self.bus.publish(
                    {"ts": _now(), "type": "run_end", "final_answer": final[:500]}
                )
        except Exception as exc:  # noqa: BLE001
            self.trace.record("error", detail=f"{type(exc).__name__}: {exc}")
            self.bus.publish({"ts": _now(), "type": "run_end", "final_answer": ""})

    def _emit_status(self, merged: dict[str, Any], node_name: str) -> None:
        memory = build_layered_memory(merged, node=node_name)
        todos = merged.get("todos", []) or []
        done = sum(1 for t in todos if t.get("status") == "completed")
        self.bus.publish(
            {
                "ts": _now(),
                "type": "status",
                "node": node_name,
                "task": merged.get("task", ""),
                "todo_done": done,
                "todo_total": len(todos),
                "attempts": merged.get("attempts", 0),
                "passed": merged.get("passed"),
                "handoffs": len(merged.get("agent_handoffs", []) or []),
                "sources": len(merged.get("sources", []) or []),
                "memory": memory_event(memory, node=node_name),
            }
        )
