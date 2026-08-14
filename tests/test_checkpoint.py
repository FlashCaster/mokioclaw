"""M7 第一阶段测试：检查点（不依赖 LLM）。"""

from mokioclaw.core.checkpoint import (
    _snapshot_files,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from mokioclaw.core.state import RuntimeState


def _graph_state():
    return {
        "task": "写一个 hello.py 并验证",
        "plan_summary": "写文件然后跑验证",
        "todos": [{"id": "todo-1", "content": "写 hello.py", "status": "completed", "note": ""}],
        "acceptance_criteria": ["文件存在"],
        "verification_commands": ["python hello.py"],
        "attempts": 1,
        "passed": True,
        "verifier_summary": "ok",
        "agent_handoffs": [{"from_agent": "planner", "to_agent": "codeAgent"}],
        "messages": [],  # 不可序列化字段，应被排除出检查点
    }


# ---- 模式 ----

def test_off_mode_skips(tmp_path):
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="off")
    result = save_checkpoint(runtime, {"task": "x"}, node="planner")
    assert result["ok"] is True
    assert result["skipped"] is True


def test_light_mode_writes_latest(tmp_path):
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    result = save_checkpoint(runtime, _graph_state(), node="verifier")
    assert result["ok"] is True
    assert result["mode"] == "light"
    ckpt_dir = tmp_path / ".mokioclaw" / "checkpoints"
    assert (ckpt_dir / "state.json").exists()
    assert (ckpt_dir / "RECOVERY.md").exists()
    assert (ckpt_dir / "files.json").exists()


def test_strict_mode_keeps_history(tmp_path):
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="strict")
    save_checkpoint(runtime, _graph_state(), node="planner")
    save_checkpoint(runtime, _graph_state(), node="verifier")
    listed = list_checkpoints(runtime)
    assert listed["count"] == 2
    ckpt_dir = tmp_path / ".mokioclaw" / "checkpoints"
    assert (ckpt_dir / "ckpt-0001" / "state.json").exists()
    assert (ckpt_dir / "ckpt-0002" / "state.json").exists()


def test_invalid_mode_rejected(tmp_path):
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="bogus")
    result = save_checkpoint(runtime, {"task": "x"}, node="planner")
    assert result["ok"] is False
    assert "invalid checkpoint mode" in result["error"]


# ---- 恢复 ----

def test_load_checkpoint_roundtrip(tmp_path):
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    save_checkpoint(runtime, _graph_state(), node="final")
    loaded = load_checkpoint(runtime)
    assert loaded["ok"] is True
    assert loaded["exists"] is True
    assert loaded["state"]["task"] == "写一个 hello.py 并验证"
    assert loaded["state"]["todos"][0]["id"] == "todo-1"
    # messages 不应进入检查点
    assert "messages" not in loaded["state"]


def test_load_checkpoint_missing(tmp_path):
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    loaded = load_checkpoint(runtime)
    assert loaded["ok"] is False
    assert loaded["exists"] is False


# ---- 文件快照 ----

def test_snapshot_excludes_internal_dirs(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / ".mokioclaw" / "x").mkdir(parents=True)
    (tmp_path / ".mokioclaw" / "x" / "skip.txt").write_text("x", encoding="utf-8")
    files = _snapshot_files(tmp_path)
    paths = {f["path"] for f in files}
    assert "hello.py" in paths
    assert not any(".mokioclaw" in p for p in paths)


def test_snapshot_has_sha256(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    files = _snapshot_files(tmp_path)
    assert files[0]["sha256"]
    assert len(files[0]["sha256"]) == 64
