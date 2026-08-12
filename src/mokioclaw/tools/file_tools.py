"""File tools: read, write, and edit files within the workspace sandbox."""

from pathlib import Path

from mokioclaw.core.paths import resolve_workspace_path
from mokioclaw.core.state import RuntimeState


def read_file(
    state: RuntimeState,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> dict:
    """Read a text file from the workspace.

    Args:
        file_path: Path relative to workspace root.
        offset: Line number to start reading from (0-indexed).
        limit: Maximum number of lines to read.

    Returns:
        dict with keys: ok, file_path, content, total_lines, offset, limit,
        and optionally error.
    """
    try:
        target = resolve_workspace_path(state.workspace, file_path)
        if not target.exists():
            return {
                "ok": False,
                "error": f"File not found: {file_path}",
                "file_path": file_path,
            }
        if not target.is_file():
            return {
                "ok": False,
                "error": f"Not a file: {file_path}",
                "file_path": file_path,
            }

        lines = target.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)
        sliced = lines[offset : offset + limit]

        return {
            "ok": True,
            "file_path": file_path,
            "content": "\n".join(sliced),
            "total_lines": total_lines,
            "offset": offset,
            "limit": limit,
        }
    except ValueError as e:
        return {"ok": False, "error": str(e), "file_path": file_path}
    except Exception as e:
        return {"ok": False, "error": f"Read failed: {e}", "file_path": file_path}


def write_file(
    state: RuntimeState,
    file_path: str,
    content: str,
) -> dict:
    """Create or overwrite a file in the workspace.

    Args:
        file_path: Path relative to workspace root.
        content: Full text content to write.

    Returns:
        dict with keys: ok, file_path, bytes_written, and optionally error.
    """
    try:
        target = resolve_workspace_path(state.workspace, file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        size = len(content.encode("utf-8"))
        return {
            "ok": True,
            "file_path": file_path,
            "bytes_written": size,
        }
    except ValueError as e:
        return {"ok": False, "error": str(e), "file_path": file_path}
    except Exception as e:
        return {"ok": False, "error": f"Write failed: {e}", "file_path": file_path}


def edit_file(
    state: RuntimeState,
    file_path: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Replace a unique text fragment in an existing file.

    The old_text must match exactly one location in the file.
    If it matches zero or more than one, the edit is rejected.

    Args:
        file_path: Path relative to workspace root.
        old_text: Exact text to find and replace.
        new_text: Replacement text.

    Returns:
        dict with keys: ok, file_path, replacements, and optionally error.
    """
    try:
        target = resolve_workspace_path(state.workspace, file_path)
        if not target.exists():
            return {
                "ok": False,
                "error": f"File not found: {file_path}",
                "file_path": file_path,
            }

        current = target.read_text(encoding="utf-8")
        count = current.count(old_text)

        if count == 0:
            return {
                "ok": False,
                "error": f"old_text not found in {file_path}",
                "file_path": file_path,
            }
        if count > 1:
            return {
                "ok": False,
                "error": (
                    f"old_text matches {count} locations in {file_path}. "
                    f"Make it more specific so it matches exactly once."
                ),
                "file_path": file_path,
                "match_count": count,
            }

        new_content = current.replace(old_text, new_text, 1)
        target.write_text(new_content, encoding="utf-8")

        return {
            "ok": True,
            "file_path": file_path,
            "replacements": 1,
        }
    except ValueError as e:
        return {"ok": False, "error": str(e), "file_path": file_path}
    except Exception as e:
        return {"ok": False, "error": f"Edit failed: {e}", "file_path": file_path}
