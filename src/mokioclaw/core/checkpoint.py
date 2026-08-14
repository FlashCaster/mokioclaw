"""Checkpoint & resume（M7 第一阶段：检查点）。

三种模式（对应 Stage 5 Day 14 的 light/strict/off）：
  - off   ：不保存检查点（默认行为不变）
  - light ：每次检查点覆盖写 state.json + RECOVERY.md + files.json，只保留「最新可恢复状态」
  - strict：light 基础上，每次检查点保留独立快照目录 ckpt-<seq>/ + manifest 累积，
            并尽力用内部 git（分离 work-tree）记录文件快照，git 不可用时降级为纯文件快照。

存储位置：<workspace>/.mokioclaw/checkpoints/
  light  模式下直接是 state.json / RECOVERY.md / files.json（覆盖）
  strict 模式下是 ckpt-0001/ ... ckpt-000N/ 目录 + manifest.json + git/

恢复（resume）：读最新检查点的 state.json，恢复关键图状态（task/todos/plan/attempts/
passed/handoffs 等）。完整消息流不恢复——三层记忆（TODO.md / NOTEPAD.md /
HISTORY_SUMMARY.md）已在工作区内持久化，agent 可据此恢复上下文。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from mokioclaw.core.state import RuntimeState

CHECKPOINT_DIR = ".mokioclaw/checkpoints"
STATE_FILE = "state.json"
RECOVERY_FILE = "RECOVERY.md"
FILES_FILE = "files.json"
MANIFEST_FILE = "manifest.json"
GIT_DIR_NAME = "git"

VALID_MODES = {"off", "light", "strict"}

# 文件快照要排除的目录（避免把检查点自身 / 虚拟环境 / 缓存纳入快照）
EXCLUDE_DIRS = {".mokioclaw", ".git", "__pycache__", ".venv", ".pytest_cache", "node_modules"}

# 可序列化进检查点的图状态字段（排除 messages / runtime 等不可 JSON 化的字段）
_SERIALIZABLE_KEYS = (
    "task",
    "plan_summary",
    "todos",
    "acceptance_criteria",
    "verification_commands",
    "attempts",
    "max_attempts",
    "passed",
    "verifier_summary",
    "verification_checks",
    "recommended_next_instruction",
    "research_notes",
    "sources",
    "agent_handoffs",
    "code_agent_summary",
    "context_summary",
    "context_token_count",
    "history_summary",
    "compression_events",
    "last_error",
    "final_answer",
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _checkpoint_root(workspace: Path) -> Path:
    return Path(workspace) / ".mokioclaw" / "checkpoints"


def _serializable_state(state: dict[str, Any]) -> dict[str, Any]:
    """从图状态挑选可持久化的关键字段。"""
    return {key: state.get(key) for key in _SERIALIZABLE_KEYS}


def _snapshot_files(workspace: Path) -> list[dict[str, Any]]:
    """对 workspace 内文件做 sha256 快照（排除 .mokioclaw / .git / venv 等）。"""
    snapshots: list[dict[str, Any]] = []
    root = Path(workspace)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            digest = ""
        snapshots.append(
            {
                "path": rel.as_posix(),
                "sha256": digest,
                "size": path.stat().st_size,
            }
        )
    return snapshots


def render_recovery_markdown(meta: dict[str, Any], state: dict[str, Any]) -> str:
    """渲染人类可读的恢复摘要 RECOVERY.md。"""
    lines = [
        "# MokioClaw 恢复摘要",
        "",
        f"- 检查点：{meta.get('id', '')}",
        f"- 时间：{meta.get('ts', '')}",
        f"- 节点：{meta.get('node', '')}",
        f"- 模式：{meta.get('mode', '')}",
        "",
        "## 任务",
        "",
        str(state.get("task", "(无)")),
        "",
        "## 计划",
        "",
        str(state.get("plan_summary", "(无)")),
        "",
        "## 进度（Todos）",
        "",
    ]
    todos = state.get("todos") or []
    if todos:
        for todo in todos:
            lines.append(
                f"- [{todo.get('status', 'pending')}] {todo.get('id', '')} {todo.get('content', '')}"
            )
    else:
        lines.append("- (无)")
    lines += [
        "",
        "## 验证",
        "",
        f"- passed: {state.get('passed')}",
        f"- 摘要: {state.get('verifier_summary', '(无)')}",
        f"- 下一步: {state.get('recommended_next_instruction', '(无)')}",
        "",
        "## 恢复方法",
        "",
        "运行 `mokioclaw --resume` 从本检查点恢复关键状态（todos/plan/attempts 等）；",
        "工作区内 TODO.md / NOTEPAD.md / HISTORY_SUMMARY.md 已持久化上下文。",
        "",
    ]
    return "\n".join(lines)


def _next_seq(root: Path) -> int:
    """根据已存在的 ckpt-<seq> 目录推断下一个序号。"""
    seqs = []
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("ckpt-"):
                try:
                    seqs.append(int(child.name.split("-")[1]))
                except (IndexError, ValueError):
                    continue
    return (max(seqs) + 1) if seqs else 1


def _git_commit(workspace: Path, git_dir: Path, message: str) -> dict[str, Any]:
    """strict 模式：分离 work-tree 的内部 git 提交（尽力而为，失败降级）。"""
    def run(args: list[str]) -> None:
        subprocess.run(
            ["git", f"--git-dir={git_dir}", f"--work-tree={workspace}", *args],
            check=True,
            capture_output=True,
            text=True,
        )

    try:
        git_dir.mkdir(parents=True, exist_ok=True)
        if not (git_dir / "HEAD").exists():
            run(["init", "--quiet"])
            run(["config", "user.name", "mokioclaw"])
            run(["config", "user.email", "mokioclaw@local"])
        # 排除 .mokioclaw 目录，避免自引用
        run(["add", "-A", "--", "."])
        run(["reset", "--", ".mokioclaw"])
        run(["commit", "--quiet", "-m", message])
        return {"ok": True, "message": message}
    except Exception as exc:
        return {"ok": False, "error": f"git commit failed: {exc}"}


def save_checkpoint(
    runtime: RuntimeState,
    state: dict[str, Any],
    *,
    node: str = "",
    mode: str | None = None,
) -> dict[str, Any]:
    """保存检查点。返回 {ok, id, mode, files, ...}。"""
    workspace = runtime.workspace
    effective_mode = mode or runtime.checkpoint_mode
    if effective_mode == "off":
        return {"ok": True, "mode": "off", "skipped": True}

    if effective_mode not in VALID_MODES:
        return {"ok": False, "error": f"invalid checkpoint mode: {effective_mode}"}

    root = _checkpoint_root(workspace)
    serializable = _serializable_state(state)
    files = _snapshot_files(workspace)

    if effective_mode == "strict":
        seq = _next_seq(root)
        ckpt_dir = root / f"ckpt-{seq:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": f"ckpt-{seq:04d}",
            "ts": _now(),
            "iso": _now_iso(),
            "node": node,
            "mode": effective_mode,
            "file_count": len(files),
        }
        state_payload = {**meta, "state": serializable}
        (ckpt_dir / STATE_FILE).write_text(
            json.dumps(state_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (ckpt_dir / FILES_FILE).write_text(
            json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (ckpt_dir / RECOVERY_FILE).write_text(
            render_recovery_markdown(meta, serializable), encoding="utf-8"
        )
        # 累积 manifest
        manifest = _read_manifest(root)
        manifest["checkpoints"].append(meta)
        _write_manifest(root, manifest)
        # 尽力内部 git
        git_result = _git_commit(workspace, root / GIT_DIR_NAME, f"checkpoint {meta['id']} ({node})")
        return {
            "ok": True,
            "id": meta["id"],
            "mode": effective_mode,
            "dir": str(ckpt_dir),
            "files": len(files),
            "git": git_result.get("ok"),
            "state": serializable,
        }

    # light 模式：覆盖最新
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "latest",
        "ts": _now(),
        "iso": _now_iso(),
        "node": node,
        "mode": effective_mode,
        "file_count": len(files),
    }
    state_payload = {**meta, "state": serializable}
    (root / STATE_FILE).write_text(
        json.dumps(state_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (root / FILES_FILE).write_text(
        json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / RECOVERY_FILE).write_text(
        render_recovery_markdown(meta, serializable), encoding="utf-8"
    )
    return {
        "ok": True,
        "id": "latest",
        "mode": effective_mode,
        "dir": str(root),
        "files": len(files),
        "state": serializable,
    }


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_FILE
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"checkpoints": []}


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_checkpoint(runtime: RuntimeState) -> dict[str, Any]:
    """读取最新检查点，返回可恢复的关键状态。"""
    root = _checkpoint_root(runtime.workspace)
    mode = runtime.checkpoint_mode

    # strict：找 manifest 里最后一个；light：读 state.json
    if mode == "strict":
        manifest = _read_manifest(root)
        ckpts = manifest.get("checkpoints", [])
        if not ckpts:
            return {"ok": False, "error": "no checkpoints found", "exists": False}
        latest = ckpts[-1]
        state_path = root / latest["id"] / STATE_FILE
    else:
        state_path = root / STATE_FILE

    if not state_path.exists():
        return {"ok": False, "error": "no checkpoint state found", "exists": False}

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid checkpoint state: {exc}", "exists": False}

    recovery_path = root / "RECOVERY.md" if mode != "strict" else state_path.parent / RECOVERY_FILE
    recovery = recovery_path.read_text(encoding="utf-8") if recovery_path.exists() else ""

    return {
        "ok": True,
        "exists": True,
        "id": payload.get("id", ""),
        "ts": payload.get("ts", ""),
        "node": payload.get("node", ""),
        "state": payload.get("state", {}),
        "recovery": recovery,
    }


def list_checkpoints(runtime: RuntimeState) -> dict[str, Any]:
    """列出所有检查点（strict 模式有历史；light 模式只有 latest）。"""
    root = _checkpoint_root(runtime.workspace)
    if not root.exists():
        return {"ok": True, "checkpoints": [], "count": 0}

    if runtime.checkpoint_mode == "strict":
        manifest = _read_manifest(root)
        checkpoints = manifest.get("checkpoints", [])
    else:
        state_path = root / STATE_FILE
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                checkpoints = [
                    {
                        "id": payload.get("id", "latest"),
                        "ts": payload.get("ts", ""),
                        "node": payload.get("node", ""),
                    }
                ]
            except json.JSONDecodeError:
                checkpoints = []
        else:
            checkpoints = []

    return {"ok": True, "checkpoints": checkpoints, "count": len(checkpoints)}
