"""CLI entry point — typer app with workspace creation and rich output."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mokioclaw.core.agent import stream_agent_events
from mokioclaw.core.paths import create_workspace
from mokioclaw.core.state import RuntimeState
from mokioclaw.tools.registry import build_tools

app = typer.Typer(
    name="mokioclaw",
    help="MokioClaw Stage 1 — ReAct minimal CodeAgent",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def callback(
    task: Optional[str] = typer.Argument(None, help="Task description for the agent"),
    workspace: Optional[Path] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="Workspace directory (auto-created if not provided)",
    ),
    approval_mode: str = typer.Option(
        "inline",
        "--approval-mode",
        help="Approval mode for high-risk bash commands: inline / deny / auto"
    )
) -> None:
    """Run the MokioClaw ReAct agent on a task.

    If no task is given, shows help and available subcommands.
    """
    if task is None:
        app.get_command(None)  # ensure subcommands registered
        raise typer.Exit()

    ws = workspace.resolve() if workspace else create_workspace()
    console.print(
        Panel.fit(f"[bold cyan]{ws}[/bold cyan]", title="MokioClaw Stage 1 · Workspace")
    )
    console.print(f"[dim]Task:[/dim] {task}\n")

    state = RuntimeState(workspace=ws, approval_mode=approval_mode)
    tools = build_tools(state)

    # Show available tools
    tool_table = Table(title="Available Tools", show_header=False)
    for t in tools:
        desc = (t.description or "")[:80]
        tool_table.add_row(f"[cyan]{t.name}[/cyan]", f"[dim]{desc}[/dim]")
    console.print(tool_table)
    console.print()

    # Run ReAct loop
    for event in stream_agent_events(task, state):
        etype = event["type"]

        if etype == "ai_message":
            loop = event.get("loop", "?")
            text = event.get("content", "")
            if text:
                console.print(f"[bold yellow]💭 Think #{loop}[/bold yellow] [dim]{text[:200]}[/dim]")

        elif etype == "tool_call":
            args_str = ", ".join(
                f"{k}={repr(v)[:60]}" for k, v in event["args"].items()
            )
            console.print(f"  [bold green]🔧 {event['name']}[/bold green]({args_str})")

        elif etype == "tool_result":
            result = event["result"]
            ok = result.get("ok", False)
            icon = "✅" if ok else "❌"
            detail = ""
            if ok:
                if "bytes_written" in result:
                    detail = f"wrote {result['bytes_written']}B"
                elif "exit_code" in result:
                    stdout_preview = result.get("stdout", "")[:80].replace("\n", " ")
                    detail = f"exit={result['exit_code']} | {stdout_preview}"
                elif "match_count" in result:
                    detail = f"{result['match_count']} matches"
            else:
                detail = result.get("error", "failed")[:100]
            console.print(f"    {icon} [dim]{detail}[/dim]")

        elif etype == "final_answer":
            console.print()
            console.print(
                Panel(
                    event["content"][:1000],
                    title="[bold green]✅ Final Answer[/bold green]",
                    border_style="green",
                )
            )

        elif etype == "error":
            console.print(f"\n[bold red]❌ {event['message']}[/bold red]")


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
