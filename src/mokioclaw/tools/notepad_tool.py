"""Notepad tool: durable notes that survive context compression.

NOTEPAD.md is the agent's "external brain" — key findings, decisions, blockers
written by the actor, read back after compression.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mokioclaw.core.paths import resolve_workspace_path
from mokioclaw.core.state import RuntimeState

NOTEPAD_FILE = "NOTEPAD.md"


def read_notepad(state: RuntimeState) -> dict[str, Any]:
    try:
        target = resolve_workspace_path(state.workspace, NOTEPAD_FILE)
        if not target.exists():
            return {"ok": True, "path": NOTEPAD_FILE, "content": "", "exists": False}
        content = target.read_text(encoding="utf-8")
        return {"ok": True, "path": NOTEPAD_FILE, "content": content, "exists": True}
    except ValueError as e:
        return {"ok": False, "error": str(e), "exists": False}
    except Exception as e:
        return {"ok": False, "error": f"read notepad failed: {e}", "exists": False}


def append_notepad(state: RuntimeState, heading: str, content: str) -> dict[str, Any]:
    if not content.strip():
        return {"ok": False, "error": "content must not be empty"}
    try:
        target = resolve_workspace_path(state.workspace, NOTEPAD_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8") if target.exists() else "# MokioClaw Notepad\n"
        title = heading.strip() or "Note"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n## {title}\n\n_Recorded: {timestamp}_\n\n{content.strip()}\n"
        updated = existing.rstrip() + "\n" + entry
        target.write_text(updated, encoding="utf-8")
        return {"ok": True, "path": NOTEPAD_FILE, "heading": title, "lines": len(updated.splitlines())}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"append notepad failed: {e}"}
