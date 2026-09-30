"""进程内结果推送：订阅 / 发布。

订阅发生在 HTTP 协程（绑定了 asyncio 事件循环）里；估计发生在普通
工作线程里。发布时用创建订阅时捕获的循环 ``call_soon_threadsafe``
把结果投递到每个订阅者的 ``asyncio.Queue``，因此线程安全。只实现
单进程常驻后端需要的进程内分发，不依赖外部消息中间件。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass(eq=False)
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: "asyncio.Queue[dict]"
    last_version: int


class PubSub:
    def __init__(self) -> None:
        self._subs: dict[str, set[_Subscriber]] = {}

    def subscribe(self, session_id: str,
                  after_version: int = 0) -> tuple[_Subscriber, list[dict]]:
        """注册订阅，返回 (订阅句柄, 已有的比 after_version 新的结果)。

        已有结果由调用方（manager）提供；这里只登记队列。
        """
        sub = _Subscriber(
            loop=asyncio.get_running_loop(),
            queue=asyncio.Queue(maxsize=1024),
            last_version=after_version,
        )
        self._subs.setdefault(session_id, set()).add(sub)
        return sub, []

    def unsubscribe(self, session_id: str, sub: _Subscriber) -> None:
        bucket = self._subs.get(session_id)
        if bucket:
            bucket.discard(sub)
            if not bucket:
                self._subs.pop(session_id, None)

    def publish(self, session_id: str, result: dict) -> None:
        """工作线程侧调用：把一条结果推给该会话的所有订阅者。"""
        for sub in list(self._subs.get(session_id, ())):  # 发布期间可能退订
            version = int(result["version"])
            if version <= sub.last_version:
                continue
            sub.loop.call_soon_threadsafe(self._offer, sub, result)

    @staticmethod
    def _offer(sub: _Subscriber, result: dict) -> None:
        try:
            sub.queue.put_nowait(result)
            sub.last_version = int(result["version"])
        except asyncio.QueueFull:
            # 慢消费者：丢弃最旧的一条腾位，保证最新结果总能发出
            try:
                sub.queue.get_nowait()
                sub.queue.put_nowait(result)
                sub.last_version = int(result["version"])
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
