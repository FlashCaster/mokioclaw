"""CLI entry point — typer app with workspace creation and rich output (M4)."""

import json
from pathlib import Path
from typing import Optional

import typer
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mokioclaw.core.paths import create_workspace
from mokioclaw.core.state import RuntimeState
from mokioclaw.graph.workflow import build_workflow
from mokioclaw.tools.registry import build_tools

app = typer.Typer(
    name="mokioclaw",
    help="MokioClaw Stage 2 — planner→actor→verifier CodeAgent",
    invoke_without_command=True,
)
console = Console()


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
    task: Optional[str] = typer.Argument(None, help="Task description for the agent"),
    workspace: Optional[Path] = typer.Option(
        None, "-w", "--workspace", help="Workspace directory (auto-created if not provided)"
    ),
    approval_mode: str = typer.Option(
        "inline", "--approval-mode", help="Approval mode for high-risk bash: inline / deny / auto"
    ),
) -> None:
    """Run the MokioClaw planner→actor→verifier agent on a task."""
    if task is None:
        app.get_command(None)
        raise typer.Exit()

    ws = workspace.resolve() if workspace else create_workspace()
    console.print(Panel.fit(f"[bold cyan]{ws}[/bold cyan]", title="MokioClaw Stage 2 · Workspace"))
    console.print(f"[dim]Task:[/dim] {task}\n")

    state = RuntimeState(workspace=ws, approval_mode=approval_mode)

    graph = build_workflow()
    initial = {"task": task, "runtime": state, "max_attempts": 3, "attempts": 0}

    for step in graph.stream(initial):
        for node_name, data in step.items():
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


if __name__ == "__main__":
    app()
