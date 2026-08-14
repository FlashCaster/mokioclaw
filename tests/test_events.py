"""M7c 测试：事件总线 + 审批桥（不依赖 LLM / TUI）。"""

import threading
import time

from mokioclaw.core.events import ApprovalBridge, EventBus


# ---- EventBus ----

def test_event_bus_publish_and_drain():
    bus = EventBus()
    bus.publish({"type": "node_start", "node": "planner"})
    bus.publish({"type": "node_end", "node": "planner"})
    events = bus.drain()
    assert len(events) == 2
    assert events[0]["type"] == "node_start"
    # drain 后队列清空
    assert bus.drain() == []


def test_event_bus_empty():
    bus = EventBus()
    assert bus.is_empty() is True
    assert bus.drain() == []


# ---- ApprovalBridge ----

def test_approval_bridge_roundtrip():
    bridge = ApprovalBridge(timeout=5)
    result = {}

    def requester():
        result["approved"] = bridge.request("pip install requests")

    thread = threading.Thread(target=requester, daemon=True)
    thread.start()
    time.sleep(0.2)  # 等请求入队

    command = bridge.poll_request()
    assert command == "pip install requests"
    bridge.resolve(command, True)
    thread.join(timeout=2)
    assert result["approved"] is True


def test_approval_bridge_deny_roundtrip():
    bridge = ApprovalBridge(timeout=5)
    result = {}

    def requester():
        result["approved"] = bridge.request("rm -rf /tmp/x")

    thread = threading.Thread(target=requester, daemon=True)
    thread.start()
    time.sleep(0.2)

    command = bridge.poll_request()
    bridge.resolve(command, False)
    thread.join(timeout=2)
    assert result["approved"] is False


def test_approval_bridge_timeout_defaults_deny():
    bridge = ApprovalBridge(timeout=0.3)
    # 无人 resolve，超时后默认拒绝（安全优先）
    assert bridge.request("curl http://x") is False


def test_approval_bridge_cancel_all():
    bridge = ApprovalBridge(timeout=5)
    results = {}

    def requester():
        results["approved"] = bridge.request("pip install requests")

    thread = threading.Thread(target=requester, daemon=True)
    thread.start()
    time.sleep(0.2)

    assert bridge.poll_request() == "pip install requests"
    bridge.cancel_all()  # 停止任务时清理，旧线程应立即返回 False
    thread.join(timeout=2)
    assert results["approved"] is False
