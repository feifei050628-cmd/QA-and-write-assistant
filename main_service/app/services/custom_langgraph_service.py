"""调用自定义 LangGraph FastAPI 业务接口的 HTTP 客户端。

这里刻意不导入 ``langgraph_sdk``。客户端只认识本项目定义的 JSON 和 SSE，
因此 LangGraph 服务可以由普通 Uvicorn/Gunicorn worker 部署，不需要购买或
运行 LangGraph Server。
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from config.settings import Settings


class CustomLangGraphService:
    """面向业务的 LangGraph HTTP 客户端。

    它是 main_service 与 langgraph_service 之间的协议适配层：上层传入
    ``query``，下游收到 ``message``；流式接口直接返回下游的 SSE 文本，
    不在 main_service 做事件解析或字段转换。
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or Settings.LANGGRAPH_API_URL).rstrip("/")
        # 普通 JSON 请求必须有读取上限；只有 SSE 流式请求允许长期等待。
        # 如果把 read=None 复用于历史接口，下游连接异常时页面会永久 pending。
        self.timeout = httpx.Timeout(connect=10.0, read=20.0, write=30.0, pool=10.0)
        # 复用 HTTP 连接，避免每次读取历史都重新创建 TCP 连接和连接池。
        # 该客户端在 main_service 进程生命周期内保持，服务关闭时由 lifespan
        # 调用 close() 释放连接。
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        """预热到 langgraph_service 的 HTTP 客户端。"""

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def shutdown(self) -> None:
        """关闭复用的 HTTP 客户端。"""

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取进程级复用客户端，兼容未经过 lifespan 的单元调用。"""

        if self._client is None:
            await self.startup()
        assert self._client is not None
        return self._client

    def _payload(self, *, thread_id: str, conversation_id: str | None, query: str, knowledge_bool: bool | None) -> dict[str, Any]:
        """把 main_service 的参数转换为自定义服务请求体。"""

        # main_service 对外使用 query 字段，而 langgraph_service 的业务模型
        # 使用 message 字段。这里是两个服务之间唯一的请求字段适配点，避免
        # 把内部字段名泄漏到前端，也避免在每个路由里重复组装 JSON。
        payload: dict[str, Any] = {
            "user_id": str(thread_id),
            "conversation_id": conversation_id,
            "message": query,
        }
        # None 表示调用方没有指定知识库策略。省略该字段可以让图使用自己的
        # 默认路由；只有明确传入 True/False 时才覆盖图的默认行为。
        if knowledge_bool is not None:
            payload["knowledge_bool"] = knowledge_bool
        return payload

    async def chat(self, *, thread_id: str, conversation_id: str | None = None, query: str, knowledge_bool: bool | None = None) -> dict[str, Any]:
        """调用自定义非流式聊天接口。"""

        client = await self._get_client()
        response = await client.post(f"{self.base_url}/api/chat", json=self._payload(thread_id=thread_id, conversation_id=conversation_id, query=query, knowledge_bool=knowledge_bool))
        response.raise_for_status()
        return response.json()

    async def chat_stream(self, *, thread_id: str, conversation_id: str | None = None, query: str, knowledge_bool: bool | None = None) -> AsyncIterator[str]:
        """逐行返回下游 SSE，保持事件内容不变。"""

        # 流式请求不能使用普通的短 read timeout，否则模型长时间生成时会被
        # httpx 误判为超时；类初始化时已将 read timeout 设为 None。
        client = await self._get_client()
        # read=None 表示不限制响应读取时间。流式模型可能持续输出较长时间，
        # 如果沿用普通请求的 read timeout，客户端会在模型尚未结束时误判超时。
        stream_timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        async with client.stream("POST", f"{self.base_url}/api/chat/stream", json=self._payload(thread_id=thread_id, conversation_id=conversation_id, query=query, knowledge_bool=knowledge_bool), timeout=stream_timeout) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                # 保留 SSE 的所有行和空行，由调用方原样写回客户端。
                yield line

    async def get_history(self, *, thread_id: str, conversation_id: str | None = None) -> list[dict[str, Any]]:
        """读取自定义服务返回的消息列表。"""

        if not conversation_id:
            return []
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/api/conversations/{conversation_id}/history", params={"user_id": str(thread_id)})
        response.raise_for_status()
        return response.json().get("messages", [])

    async def clear_history(self, *, thread_id: str, conversation_id: str | None = None) -> None:
        """删除指定会话的短期 checkpoint。"""

        if not conversation_id:
            return
        client = await self._get_client()
        response = await client.delete(f"{self.base_url}/api/conversations/{conversation_id}", params={"user_id": str(thread_id)})
        response.raise_for_status()

    async def cancel_runs(self, *, thread_id: str, conversation_id: str | None = None, action: str = "interrupt") -> None:
        """请求自定义服务取消当前会话任务。action 保留用于兼容旧调用方。"""

        del action
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/api/chat/cancel", json={"user_id": str(thread_id), "conversation_id": conversation_id})
        response.raise_for_status()

    async def resume_stream(self, *, thread_id: str, conversation_id: str | None = None, action: str, content: str | None = None, answer: str | None = None) -> AsyncIterator[str]:
        """提交 interrupt 恢复参数并逐行返回下游 SSE。"""

        # action 由前端审阅面板产生：approve 接受草稿，edit 提交修改内容，
        # rewrite 要求重新生成，clarify 则把 answer 作为澄清结果恢复图执行。
        # 恢复请求必须使用原 conversation_id，否则下游会计算出新的 thread，
        # 找不到保存 interrupt 的 checkpoint。
        payload = {"user_id": str(thread_id), "conversation_id": conversation_id, "action": action, "content": content, "answer": answer}
        client = await self._get_client()
        async with client.stream("POST", f"{self.base_url}/api/chat/resume", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line
