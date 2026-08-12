"""Grep tool: regex search within workspace files."""

import fnmatch
import re
from pathlib import Path

from mokioclaw.core.state import RuntimeState


def grep(
    state: RuntimeState,
    pattern: str,
    path: str = ".",
    glob: str = "*",
    head_limit: int = 20,
    ignore_case: bool = True,
) -> dict:
    """Search for a regex pattern in workspace files.

    Args:
        pattern: Python regex pattern to search for.
        path: Subdirectory or file path relative to workspace root.
              Use "." to search entire workspace.
        glob: Filename glob filter (e.g. "*.py", "*.md").
        head_limit: Max number of matching lines to return.
        ignore_case: Whether to ignore case (default True).

    Returns:
        dict with keys: ok, pattern, matches, match_count, truncated.
    """
    try:
        # Resolve search root
        if path == ".":
            search_root = state.workspace
        else:
            # Simple resolution without full sandbox check for subdirectories
            clean = path.lstrip("/").lstrip("\\")
            search_root = (state.workspace / clean).resolve()
            try:
                search_root.relative_to(state.workspace.resolve())
            except ValueError:
                return {
                    "ok": False,
                    "error": f"Path escapes workspace: {path}",
                    "pattern": pattern,
                }

        if not search_root.exists():
            return {
                "ok": False,
                "error": f"Path not found: {path}",
                "pattern": pattern,
            }

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {
                "ok": False,
                "error": f"Invalid regex: {e}",
                "pattern": pattern,
            }

        matches: list[dict] = []

        if search_root.is_file():
            files = [search_root]
        else:
            files = [
                p
                for p in search_root.rglob("*")
                if p.is_file() and fnmatch.fnmatch(p.name, glob)
            ]

        for f in files:
            if len(matches) >= head_limit:
                break
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if len(matches) >= head_limit:
                        break
                    if regex.search(line):
                        rel = f.relative_to(state.workspace)
                        matches.append(
                            {
                                "file": str(rel),
                                "line_num": i,
                                "line": line.strip(),
                            }
                        )
            except (UnicodeDecodeError, OSError):
                continue

        truncated = len(matches) >= head_limit
        return {
            "ok": True,
            "pattern": pattern,
            "matches": matches,
            "match_count": len(matches),
            "truncated": truncated,
        }
    except Exception as e:
        return {"ok": False, "error": f"Grep failed: {e}", "pattern": pattern}
