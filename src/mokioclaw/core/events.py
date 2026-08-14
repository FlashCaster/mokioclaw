"""事件总线 + 审批桥（M7c TUI 的线程通信基础设施）。

agent 在后台线程同步运行（graph.stream / LLM 调用），TUI 在 asyncio 事件循环里
渲染。两者通过线程安全的队列通信：

  - EventBus       ：agent 线程 publish 事件，TUI 线程 drain 消费
  - ApprovalBridge ：agent 线程发起审批请求并阻塞，TUI 线程弹窗回传决定
"""

from __future__ import annotations

import queue
import threading
from typing import Any


class EventBus:
    """线程安全事件总线（单向：生产者 → 消费者）。"""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()

    def publish(self, event: dict[str, Any]) -> None:
        """生产者（agent 线程）发布事件。"""
        self._queue.put(event)

    def drain(self) -> list[dict[str, Any]]:
        """消费者（TUI 线程）非阻塞取出所有待处理事件。"""
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def is_empty(self) -> bool:
        return self._queue.empty()


class ApprovalBridge:
    """agent 线程 ↔ TUI 线程的审批通信桥（Human-in-the-loop）。

    agent 线程调用 request(command) 阻塞等待；TUI 线程弹出审批 Modal 后调用
    resolve(command, approved) 回传决定。按 command 字符串作为请求 key。
    """

    def __init__(self, timeout: float = 600.0) -> None:
        self.requests: queue.Queue = queue.Queue()
        self._pending: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self.timeout = timeout

    def request(self, command: str) -> bool:
        """agent 线程调用：发起审批请求，阻塞直到 TUI 回传决定。

        超时 / 异常时默认拒绝（安全优先）。
        """
        result_q: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[command] = result_q
        self.requests.put(command)
        try:
            return bool(result_q.get(timeout=self.timeout))
        except queue.Empty:
            return False

    def resolve(self, command: str, approved: bool) -> bool:
        """TUI 线程调用：回传审批决定。返回是否成功找到对应请求。"""
        with self._lock:
            result_q = self._pending.pop(command, None)
        if result_q is None:
            return False
        result_q.put(bool(approved))
        return True

    def poll_request(self) -> str | None:
        """TUI 线程调用：非阻塞取出一个待审批命令（无则返回 None）。"""
        try:
            return self.requests.get_nowait()
        except queue.Empty:
            return None
