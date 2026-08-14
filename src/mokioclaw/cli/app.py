"""CLI entry point — typer app with workspace creation and rich output (M4)."""

import json
from pathlib import Path
from typing import Optional

import typer
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mokioclaw.core.checkpoint import load_checkpoint, save_checkpoint
from mokioclaw.core.paths import create_workspace
from mokioclaw.core.state import RuntimeState
from mokioclaw.core.trace import TraceWriter
from mokioclaw.graph.workflow import build_entry_workflow, build_workflow
from mokioclaw.tools.registry import build_tools

app = typer.Typer(
    name="mokioclaw",
    help="MokioClaw — planner→verifier multi-agent CodeAgent",
    invoke_without_command=True,
)
console = Console()


def _find_latest_workspace(base_dir: Path) -> "Path | None":
    """在 base_dir/.mokioclaw/workspaces/ 下找最近修改的 workspace（供 --resume 使用）。"""
    workspaces_dir = base_dir / ".mokioclaw" / "workspaces"
    if not workspaces_dir.exists():
        return None
    dirs = [d for d in workspaces_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _render_messages(messages) -> None:
    """渲染节点产出的消息：AI 文本 + 工具调用 + 工具结果。"""
    for msg in messages:
        if isinstance(msg, AIMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    args_str = ", ".join(
                        f"{k}={repr(v)[:50]}" for k, v in tc.get("args", {}).items()
                    )
                    console.print(f"  [bold green]🔧 {tc.get('name')}[/bold green]({args_str})")
            elif text:
                console.print(f"  [dim]💭 {text[:150]}[/dim]")
        elif isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content)
            except Exception:
                result = {"raw": str(msg.content)[:80]}
            ok = result.get("ok", False)
            icon = "✅" if ok else "❌"
            detail = ""
            if ok:
                detail = (
                    str(result.get("path", ""))
                    or str(result.get("stdout", ""))[:60].replace("\n", " ")
                    or str(result.get("exit_code", ""))
                )
            else:
                detail = str(result.get("error", ""))[:80]
            console.print(f"    {icon} [dim]{detail}[/dim]")


@app.callback()
def callback(
    ctx: typer.Context,
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="Task description for the agent (or use a subcommand: tui / list-tools)"
    ),
    workspace: Optional[Path] = typer.Option(
        None, "-w", "--workspace", help="Workspace directory (auto-created if not provided)"
    ),
    approval_mode: str = typer.Option(
        "inline", "--approval-mode", help="Approval mode for high-risk bash: inline / deny / auto"
    ),
    checkpoint_mode: str = typer.Option(
        "light", "--checkpoint-mode", help="Checkpoint mode: light / strict / off"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from the latest checkpoint in the workspace"
    ),
) -> None:
    """Run the MokioClaw planner→actor→verifier agent on a task."""
    # 有子命令（tui / list-tools）时，跳过 agent 逻辑，直接交给子命令执行
    if ctx.invoked_subcommand is not None:
        return
    if task is None and not resume:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    # 解析 workspace：resume 时优先用最近一次 workspace
    if workspace is not None:
        ws = workspace.resolve()
    elif resume:
        latest = _find_latest_workspace(Path.cwd())
        if latest is None:
            console.print("[red]No workspace found to resume.[/red]")
            raise typer.Exit(code=1)
        ws = latest
    else:
        ws = create_workspace()

    trace = TraceWriter(ws)
    state = RuntimeState(
        workspace=ws,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace=trace,
    )

    initial: dict = {"task": task or "", "runtime": state, "max_attempts": 3, "attempts": 0}

    # resume：从最新检查点恢复关键状态
    if resume:
        loaded = load_checkpoint(state)
        if not loaded.get("ok"):
            console.print(f"[yellow]无法恢复检查点：{loaded.get('error', '')}[/yellow]")
        else:
            recovered = loaded.get("state", {})
            for key in (
                "task", "todos", "plan_summary", "acceptance_criteria",
                "verification_commands", "attempts", "passed", "verifier_summary",
                "recommended_next_instruction", "agent_handoffs", "sources",
                "code_agent_summary", "research_notes",
            ):
                if key in recovered and recovered[key] is not None:
                    initial[key] = recovered[key]
            console.print(
                Panel(
                    loaded.get("recovery", "") or "(no recovery notes)",
                    title=f"[bold cyan]Resume · {loaded.get('id', '')} · {loaded.get('ts', '')}[/bold cyan]",
                    border_style="cyan",
                )
            )

    console.print(Panel.fit(f"[bold cyan]{ws}[/bold cyan]", title="MokioClaw · Workspace"))
    console.print(f"[dim]Task:[/dim] {initial['task']}")
    console.print(
        f"[dim]Checkpoint: {checkpoint_mode} ｜ Trace: on ｜ Approval: {approval_mode}[/dim]\n"
    )

    trace.record("run_start", task=initial["task"], checkpoint_mode=checkpoint_mode)

    # 意图路由（Stage 6）：先判断 chat / workflow
    entry_graph = build_entry_workflow()
    entry_state: dict = dict(initial)
    for step in entry_graph.stream(initial):
        for node_name, data in step.items():
            entry_state.update(data)
            trace.record("node_start", node=node_name)
            if node_name == "intent_router":
                route = data.get("intent_route", "workflow")
                console.print(
                    f"[dim]🧭 路由:[/dim] [bold]{route}[/bold] "
                    f"[dim](confidence {data.get('intent_confidence', 0):.2f})[/dim]\n"
                )
            elif node_name == "chat_responder":
                console.print(
                    Panel(
                        data.get("chat_response", ""),
                        title="[bold cyan]💬 MokioClaw[/bold cyan]",
                        border_style="cyan",
                    )
                )
            trace.record("node_end", node=node_name)

    if entry_state.get("intent_route") == "chat":
        trace.finalize(entry_state)
        return

    graph = build_workflow()
    merged_state: dict = dict(entry_state)

    for step in graph.stream(initial):
        for node_name, data in step.items():
            merged_state.update(data)
            trace.record("node_start", node=node_name)
            if node_name == "planner":
                console.print()
                lines = []
                if data.get("plan_summary"):
                    lines.append(f"[bold cyan]{data['plan_summary']}[/bold cyan]")
                if data.get("todos"):
                    lines.append("[bold]Todos:[/bold]")
                    for t in data["todos"]:
                        lines.append(f"  - [dim]{t.get('id','')}[/dim] {t.get('content','')}")
                if data.get("acceptance_criteria"):
                    lines.append("[bold]Acceptance:[/bold]")
                    for c in data["acceptance_criteria"]:
                        lines.append(f"  - {c}")
                if data.get("verification_commands"):
                    lines.append("[bold]Verification:[/bold]")
                    for c in data["verification_commands"]:
                        lines.append(f"  - [dim]{c}[/dim]")
                console.print(
                    Panel(
                        "\n".join(lines) if lines else "(empty plan)",
                        title="[bold magenta]📋 Plan[/bold magenta]",
                        border_style="magenta",
                    )
                )

            elif node_name == "actor":
                console.print()
                console.print("[bold]🎬 Actor[/bold]")
                _render_messages(data.get("messages", []))

            elif node_name == "verifier":
                console.print()
                status = "✅ PASSED" if data.get("passed") else "❌ FAILED"
                lines = [f"[bold]{status}[/bold]"]
                if data.get("verifier_summary"):
                    lines.append(data["verifier_summary"])
                if data.get("verification_checks"):
                    lines.append("[bold]Checks:[/bold]")
                    for c in data["verification_checks"]:
                        mark = "PASS" if c.get("passed") else "FAIL"
                        lines.append(f"  - {mark} {c.get('name','')}")
                if data.get("recommended_next_instruction"):
                    lines.append(f"[yellow]→ 下一步: {data['recommended_next_instruction']}[/yellow]")
                console.print(
                    Panel(
                        "\n".join(lines),
                        title="[bold blue]🔍 Verifier[/bold blue]",
                        border_style="blue",
                    )
                )

            elif node_name == "final":
                console.print()
                console.print(
                    Panel(
                        data.get("final_answer", ""),
                        title="[bold green]✅ Final Answer[/bold green]",
                        border_style="green",
                    )
                )

            # 节点结束：trace + checkpoint（旁路观测，不改变 agent 行为）
            trace.record("node_end", node=node_name)
            ckpt = save_checkpoint(state, merged_state, node=node_name)
            if ckpt.get("ok") and not ckpt.get("skipped"):
                trace.record(
                    "checkpoint",
                    node=node_name,
                    id=ckpt.get("id"),
                    mode=ckpt.get("mode"),
                )

    trace.finalize(merged_state)


@app.command()
def list_tools() -> None:
    """List all available tools with descriptions."""
    state = RuntimeState(workspace=Path.cwd(), approval_mode="inline")
    tools = build_tools(state)

    table = Table(title="MokioClaw Tools")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="yellow")
    table.add_column("Description", style="dim")

    for t in tools:
        params = ", ".join(t.args_schema.model_fields.keys()) if t.args_schema else "—"
        table.add_row(t.name, params, t.description or "—")

    console.print(table)


@app.command()
def tui(
    workspace: Optional[Path] = typer.Option(
        None, "-w", "--workspace", help="Workspace directory (auto-created if not provided)"
    ),
    approval_mode: str = typer.Option(
        "inline", "--approval-mode", help="Approval mode for high-risk bash: inline / deny / auto"
    ),
    checkpoint_mode: str = typer.Option(
        "light", "--checkpoint-mode", help="Checkpoint mode: light / strict / off"
    ),
) -> None:
    """Launch the interactive Textual TUI."""
    from mokioclaw.tui.app import MokioClawApp

    ws = workspace.resolve() if workspace else create_workspace()
    app_ui = MokioClawApp(
        workspace=ws,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
    )
    app_ui.run()


if __name__ == "__main__":
    app()
