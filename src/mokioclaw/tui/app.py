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

from rich.markup import escape

from mokioclaw.core.checkpoint import save_checkpoint
from mokioclaw.core.events import ApprovalBridge, EventBus
from mokioclaw.core.state import RuntimeState
from mokioclaw.core.trace import TraceWriter
from mokioclaw.graph.memory import build_layered_memory, memory_event
from mokioclaw.graph.workflow import build_workflow


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# 节点 → 人类可读的「正在做什么」描述（任务流程透明）
_NODE_LABEL = {
    "planner": "规划中 · 拆解任务/委派",
    "actor": "执行中 · 写代码/跑命令",
    "verifier": "验证中 · 检查结果",
    "context_monitor": "检查上下文窗口",
    "context_compressor": "压缩上下文中",
    "final": "生成最终答案",
}


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
    BINDINGS = [
        ("ctrl+x", "cancel_task", "停止任务"),
    ]
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
    #input-row {
        height: 3;
    }
    #input-row Input {
        width: 1fr;
    }
    #stop-button {
        width: 12;
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
        self._run_generation = 0  # 任务世代号：递增用于「立即取消」判断 + 事件隔离
        self._run_state = "空闲"  # 空闲 / 运行中 / 已停止 / 完成
        self._run_detail = ""
        self._last_status: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                yield RichLog(id="timeline", highlight=True, markup=True, wrap=True)
                with Horizontal(id="input-row"):
                    yield Input(placeholder="输入任务，回车执行（如：帮我创建一个贪吃蛇游戏）")
                    yield Button("⏹ 停止", id="stop-button", variant="error")
            with Vertical(id="right"):
                yield Static("等待任务…", id="status-panel")
                yield Static("Memory Snapshot 暂无", id="memory-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.15, self._poll)

    def action_cancel_task(self) -> None:
        self._cancel()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-button":
            self._cancel()

    def _cancel(self) -> None:
        if not self._running:
            self._write("[dim]没有正在运行的任务[/]")
            return
        self._running = False
        self._run_generation += 1  # 作废旧任务 → 旧线程检测到世代号不匹配即退出
        self._run_state = "已停止"
        self._run_detail = ""
        self._refresh_status()
        self._write("[yellow]⏹ 任务已停止，可直接提交新任务[/]")

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
            self._write(f"[bold yellow]审批[/] {'批准' if approved else '拒绝'}：{escape(command[:60])}")

        # 2) 事件：更新时间线 / 状态 / 记忆面板
        for event in self.bus.drain():
            self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        gen = event.get("gen")
        if gen is not None and gen != self._run_generation:
            return  # 旧任务事件，忽略
        etype = event.get("type", "")
        node = escape(str(event.get("node", "")))

        if etype == "node_start":
            label = _NODE_LABEL.get(event.get("node", ""), "")
            self._run_detail = label or f"节点 {node}"
            self._refresh_status()
            self._write(f"[cyan]▶ {node}[/] [dim]{label}[/]")
        elif etype == "node_end":
            self._write(f"[cyan]◀ 离开节点[/] [bold]{node}[/]")
        elif etype == "tool_call":
            tool = escape(str(event.get("tool", "")))
            args = escape(str(event.get("args", ""))[:80])
            self._write(f"[green]🔧 {tool}[/] {args}")
        elif etype == "tool_result":
            mark = "✅" if event.get("ok") else "❌"
            tool = escape(str(event.get("tool", "")))
            detail = escape(str(event.get("detail", ""))[:60])
            self._write(f"   {mark} {tool} {detail}")
        elif etype == "handoff":
            frm = escape(str(event.get("from_agent", "")))
            to = escape(str(event.get("to_agent", "")))
            self._write(f"[magenta]🔀 handoff[/] {frm} → {to}")
        elif etype == "checkpoint":
            cid = escape(str(event.get("id", "")))
            mode = escape(str(event.get("mode", "")))
            self._write(f"[blue]💾 检查点 {cid}[/]（{mode}）")
        elif etype == "compression":
            detail = escape(str(event.get("detail", ""))[:60])
            self._write(f"[yellow]🗜️ 上下文压缩[/] {detail}")
        elif etype == "error":
            detail = escape(str(event.get("detail", ""))[:120])
            self._write(f"[red]⚠️ 错误[/] {detail}")
        elif etype == "run_end":
            self._write("[bold green]✅ 运行结束[/]")
            self._running = False
            self._run_state = "完成"
            self._run_detail = ""
            self._refresh_status()
        elif etype == "status":
            self._update_status(event)
            self._update_memory(event)

    def _write(self, text: str) -> None:
        timeline = self.query_one("#timeline", RichLog)
        timeline.write(text)

    def _update_status(self, event: dict[str, Any]) -> None:
        self._last_status = event
        self._refresh_status()

    def _refresh_status(self) -> None:
        status = self.query_one("#status-panel", Static)
        event = self._last_status or {}
        run_icon = {
            "空闲": "⚪",
            "运行中": "🟢",
            "停止中": "🟡",
            "已停止": "⏹",
            "完成": "✅",
        }.get(self._run_state, "⚪")
        lines = [
            "[b]状态面板[/b]",
            f"运行：{run_icon} [b]{self._run_state}[/b]",
        ]
        if self._run_detail:
            lines.append(f"当前：{escape(self._run_detail)}")
        if event:
            passed = event.get("passed")
            passed_text = "✅ 通过" if passed is True else ("❌ 失败" if passed is False else "—")
            lines += [
                f"任务：{escape(str(event.get('task', ''))[:40])}",
                f"Todos：{event.get('todo_done', 0)}/{event.get('todo_total', 0)} 完成",
                f"尝试：{event.get('attempts', 0)}",
                f"验证：{passed_text}",
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
            f"node: {escape(str(event.get('node', '')))}",
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
            self._write("[yellow]⚠️ 已有任务在运行，按 Ctrl+X 停止后再提交[/]")
            return
        self._running = True
        self._run_generation += 1
        gen = self._run_generation
        self._run_state = "运行中"
        self._run_detail = "正在启动 agent..."
        self._refresh_status()
        self._write(f"[bold green]🚀 任务已提交：[/]{escape(task)}")
        threading.Thread(target=self._run_agent, args=(task, gen), daemon=True).start()

    def _run_agent(self, task: str, gen: int) -> None:
        """agent 后台线程：跑图，节点/工具事件经 TraceWriter 自动入总线。

        gen 是任务世代号：若与 self._run_generation 不一致（已被取消/被新任务取代），
        立即退出，实现「立即停止」。
        """
        try:
            graph = build_workflow()
            initial: dict[str, Any] = {
                "task": task,
                "runtime": self.state,
                "max_attempts": 3,
                "attempts": 0,
            }
            merged: dict[str, Any] = dict(initial)
            self.trace.record("run_start", task=task, checkpoint_mode=self.state.checkpoint_mode, gen=gen)

            for step in graph.stream(initial):
                if gen != self._run_generation:
                    break
                for node_name, data in step.items():
                    merged.update(data)
                    self.trace.record("node_start", node=node_name, gen=gen)
                    ckpt = save_checkpoint(self.state, merged, node=node_name)
                    if ckpt.get("ok") and not ckpt.get("skipped"):
                        self.trace.record(
                            "checkpoint",
                            node=node_name,
                            id=ckpt.get("id"),
                            mode=ckpt.get("mode"),
                            gen=gen,
                        )
                    self._emit_status(merged, node_name, gen)
                    self.trace.record("node_end", node=node_name, gen=gen)
                    if gen != self._run_generation:
                        break

            if gen != self._run_generation:
                self.trace.record("run_cancelled", reason="用户中断", gen=gen)
                self.trace.finalize(merged)
                self.bus.publish({"ts": _now(), "type": "run_cancelled", "reason": "用户中断", "gen": gen})
                return

            self.trace.finalize(merged)
            final = merged.get("final_answer", "")
            if final:
                self.bus.publish(
                    {"ts": _now(), "type": "run_end", "final_answer": final[:500], "gen": gen}
                )
        except Exception as exc:  # noqa: BLE001
            self.trace.record("error", detail=f"{type(exc).__name__}: {exc}", gen=gen)
            self.bus.publish({"ts": _now(), "type": "run_end", "final_answer": "", "gen": gen})

    def _emit_status(self, merged: dict[str, Any], node_name: str, gen: int) -> None:
        memory = build_layered_memory(merged, node=node_name)
        todos = merged.get("todos", []) or []
        done = sum(1 for t in todos if t.get("status") == "completed")
        self.bus.publish(
            {
                "ts": _now(),
                "type": "status",
                "gen": gen,
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
