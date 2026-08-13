"""Workspace path utilities — creation, resolution, safety checks."""

import os
from datetime import datetime, timezone
from pathlib import Path


def get_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


def create_workspace(base_dir: Path | None = None) -> Path:
    """Create a timestamped workspace directory.

    Args:
        base_dir: Optional parent directory. Defaults to .mokioclaw/workspaces/ under cwd.

    Returns:
        Absolute path to the new workspace.
    """
    if base_dir is None:
        base_dir = Path.cwd() / ".mokioclaw" / "workspaces"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:20]
    ws = base_dir / f"workspace-{timestamp}"
    ws.mkdir(parents=True, exist_ok=True)
    return ws.resolve()


def resolve_workspace_path(workspace: Path, file_path: str) -> Path:
    """Resolve a user-provided path relative to workspace, enforcing sandbox.

    Args:
        workspace: The workspace root directory.
        file_path: User-provided path string (relative or absolute-like).

    Returns:
        Resolved absolute path inside workspace.

    Raises:
        ValueError: If the resolved path escapes the workspace.
    """
    # Strip leading slashes/drives to force relative resolution
    clean = file_path.lstrip("/").lstrip("\\")
    # Handle Windows drive letters like "C:/..."
    if len(clean) >= 2 and clean[1] == ":":
        clean = clean[2:].lstrip("/").lstrip("\\")

    resolved = (workspace / clean).resolve()

    # Sandbox check: resolved must be inside workspace
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        raise ValueError(
            f"Path escapes workspace: '{file_path}' resolves to "
            f"'{resolved}' which is outside '{workspace.resolve()}'"
        )

    return resolved

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)