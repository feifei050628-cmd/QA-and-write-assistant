"""FastAPI 部署专用的 PostgreSQL LangGraph 运行时。

这个模块故意独立于 ``my_agent.agent``：原有 LangGraph CLI 入口继续使用原来的
图和启动方式，本 FastAPI 进程单独建立 PostgreSQL 连接，并重新构建一份图。

这里有两个不同职责的 PostgreSQL 组件：

1. ``AsyncPostgresSaver``：LangGraph checkpointer，保存 thread 级别的 checkpoint，
   也就是短期对话历史、当前图状态以及中断恢复所需的数据。
2. ``AsyncPostgresStore``：LangGraph store，保存跨 thread、按用户隔离的长期记忆。

只有把组件传给 ``builder.compile(checkpointer=..., store=...)``，图节点才能在
执行时使用它们；仅仅创建数据库连接并不会自动启用持久化。
"""

from __future__ import annotations

from contextlib import AsyncExitStack
import os
from typing import Any
from uuid import uuid4

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from config.settings import Settings
from my_agent.utils.nodes import (
    call_writing_subgraph,
    chat_agent,
    fallback_search_agent,
    finalize,
    intent_router,
    knowledge_agent,
    knowledge_guard,
    summarization_node,
)
from my_agent.utils.state import GraphState


async def load_long_term_memory(state: GraphState) -> dict[str, Any]:
    """在当前轮开始前读取用户的长期记忆。

    ``state`` 是 LangGraph 当前节点看到的完整图状态，类型标注为 ``GraphState``。
    返回值 ``dict[str, Any]`` 是“本节点对状态的增量更新”，不是要求节点返回完整
    状态；LangGraph 会把返回的字段合并回当前 state。

    ``AsyncPostgresStore`` 的 namespace 使用 ``(memories, user_id)``，因此同一个
    用户的不同 thread 可以共享长期记忆，而不同用户之间不会互相读取数据。
    读取结果被转换成 ``SystemMessage``，加入本轮图状态的 messages 中，后续摘要
    节点和 Agent 节点就可以像处理普通上下文一样使用这些记忆。

    注意：这里不把长期记忆写回 ``context``。``context`` 是原图中
    ``SummarizationNode`` 使用的短期摘要结构，直接混用会破坏原有摘要逻辑。
    """
    # user_id 来自 main_service 的业务身份，而不是本轮消息内容；只有用它
    # 作为 namespace，才能让同一用户跨会话复用记忆，同时隔离不同用户。
    user_id = str(state.get("user_id") or "default")
    # namespace 是长期记忆的逻辑分区。它与 thread_id 有意不同：thread_id
    # 只代表一个会话，user_id 让同一用户的多个会话共享可复用记忆。
    store = get_store()
    items = await store.asearch(("memories", user_id), limit=10)
    if not items:
        return {}

    lines: list[str] = []
    for item in items:
        value = getattr(item, "value", item)
        if not isinstance(value, dict):
            continue
        query = str(value.get("query") or "").strip()
        answer = str(value.get("answer") or "").strip()
        if query or answer:
            # 每条记忆保留问答边界，模型可以判断哪些历史与当前问题相关，
            # 也避免把多条记忆拼成无法区分来源的一段文本。
            lines.append(f"用户曾问：{query}\n助手曾答：{answer}")
    if not lines:
        return {}

    return {
        "messages": [
            SystemMessage(
                content="以下是该用户的长期记忆，仅在与当前问题相关时参考：\n"
                + "\n\n".join(lines)
            )
        ]
    }


async def save_long_term_memory(state: GraphState) -> dict[str, Any]:
    """在当前轮结束后保存本次问答。

    ``state`` 中的 ``messages`` 可能包含 ``HumanMessage`` 对象，也可能是从接口
    传入后暂时保留的字典，因此代码同时兼容两种格式。

    保存的数据是结构化字典，而不是拼接后的纯文本，便于后续扩展记忆类型、
    增加时间戳或按字段过滤。``index=False`` 表示当前只使用 namespace 查询，
    不要求 PostgreSQL 为这些字段建立向量/全文索引。
    """
    # user_id 由 main_service 通过输入状态传入。兜底值只用于直接调试图时，
    # 正常请求不应把所有用户都归到 default。
    user_id = str(state.get("user_id") or "default")
    query = ""
    # 从后往前寻找最近一条 HumanMessage，避免把历史问题误当成本轮 query。
    for message in reversed(state.get("messages", [])):
        message_type = getattr(message, "type", None)
        if message_type == "human" or (
            isinstance(message, dict) and (message.get("type") or message.get("role")) in {"human", "user"}
        ):
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content", "")
            query = str(content or "")
            break

    # finalize 已经把下游节点结果统一到 answer 字段，因此这里不需要再次
    # 判断当前意图或解析 AIMessage，只保存本轮最终业务答案。
    answer = str(state.get("answer") or "").strip()
    if not query and not answer:
        return {}

    # get_store() 获取的是当前 LangGraph 执行上下文中 compile(store=...) 注入的
    # store，而不是重新创建数据库连接。这样可以复用 FastAPI lifespan 管理的连接。
    store = get_store()
    # 使用随机 key 保存每轮问答，不覆盖同一用户之前的记忆；查询时由
    # asearch 取回多条记录。这里暂不启用索引，避免把普通对话文本误当成
    # 向量检索数据，后续需要语义检索时再单独设计 index 配置。
    await store.aput(
        ("memories", user_id),
        str(uuid4()),
        {
            "type": "conversation",
            "query": query,
            "answer": answer,
        },
        index=False,
    )
    return {}


def build_graph_builder() -> StateGraph:
    """构建 FastAPI 专用图，并保留原有多 Agent 拓扑。

    与原图相比，这份新增图只增加了两个边界节点：

    ``START -> load_long_term_memory``：加载跨 thread 的长期记忆；
    ``finalize -> save_long_term_memory -> END``：保存本轮问答。

    中间的 summarize、路由、知识库、写作、聊天和 finalize 节点保持原有实现，
    这样新增部署方式不会修改或影响原来的 CLI 图。
    """
    builder = StateGraph(GraphState)
    builder.add_node("load_long_term_memory", load_long_term_memory)
    builder.add_node("summarize", summarization_node)
    builder.add_node("intent_router", intent_router)
    builder.add_node("knowledge_agent", knowledge_agent)
    builder.add_node("fallback_search", fallback_search_agent)
    # 通过包装函数调用写作子图。父图只接收包装函数明确返回的
    # answer/sources/messages，不把子图的中间字段写入父图状态。
    builder.add_node("writing_workflow", call_writing_subgraph)
    builder.add_node("chat_agent", chat_agent)
    builder.add_node("finalize", finalize)
    builder.add_node("save_long_term_memory", save_long_term_memory)

    builder.add_edge(START, "load_long_term_memory")
    builder.add_edge("load_long_term_memory", "summarize")
    builder.add_edge("summarize", "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        lambda state: state["intent"],
        {
            "knowledge": "knowledge_agent",
            "writing": "writing_workflow",
            "chat": "chat_agent",
        },
    )
    builder.add_conditional_edges(
        "knowledge_agent",
        knowledge_guard,
        {
            "fallback_search": "fallback_search",
            "finalize": "finalize",
        },
    )
    builder.add_edge("fallback_search", "finalize")
    builder.add_edge("writing_workflow", "finalize")
    builder.add_edge("chat_agent", "finalize")
    builder.add_edge("finalize", "save_long_term_memory")
    builder.add_edge("save_long_term_memory", END)
    return builder


class PostgresGraphRuntime:
    """管理 FastAPI 进程生命周期内的异步 PostgreSQL 资源。

    一个运行时对象对应一个 FastAPI 进程。启动时创建并初始化 checkpointer/store，
    然后编译图；关闭时通过 ``AsyncExitStack`` 统一释放连接。路由处理函数只通过
    ``require_graph()`` 获取已经准备好的图，不直接管理数据库连接。
    """

    def __init__(self, conn_string: str | None = None) -> None:
        # 优先使用用户在根目录 .env 中配置的 POSTGRES_URI_CUSTOM。
        # 该连接串是 psycopg/LangGraph 使用的 URI 格式，而不是 SQLAlchemy URL。
        self.conn_string = (
            conn_string
            or os.getenv("POSTGRES_URI_CUSTOM")
            or Settings.LANGGRAPH_POSTGRES_DSN
        )
        self._stack = AsyncExitStack()
        self.checkpointer: AsyncPostgresSaver | None = None
        self.store: AsyncPostgresStore | None = None
        self.graph: Any = None

    async def startup(self) -> None:
        """打开连接、初始化 PostgreSQL 表结构，并编译持久化图。

        该方法由 FastAPI lifespan 在服务启动阶段调用。``from_conn_string`` 返回
        异步上下文管理器，交给 ``AsyncExitStack`` 后，服务关闭时会自动释放连接。
        ``setup`` 负责创建 LangGraph 所需表结构；最后 compile 得到的 graph 才是
        路由真正执行的对象。
        """
        # 两个对象共享同一 PostgreSQL URI，但用途不同：一个负责可恢复执行
        # 的 checkpoint，一个负责跨会话长期记忆。分别进入 AsyncExitStack，
        # shutdown 时就能成对释放资源。
        self.checkpointer = await self._stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(self.conn_string)
        )
        self.store = await self._stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(self.conn_string)
        )
        # setup() 会创建 LangGraph 所需的表。它是幂等初始化，可以在每次服务启动
        # 时执行，不需要手工先运行 SQL 脚本。
        await self.checkpointer.setup()
        await self.store.setup()

        # checkpointer 负责短期状态，store 负责长期记忆。两者必须在 compile 时
        # 注入，节点内部的 get_store() 才能取得当前执行所绑定的 store。
        # compile 是持久化真正生效的边界。若只建立连接而不传入这两个参数，
        # 图节点中的 get_store() 和 thread checkpoint 都不会连接到 PostgreSQL。
        self.graph = build_graph_builder().compile(
            checkpointer=self.checkpointer,
            store=self.store,
        )

    async def shutdown(self) -> None:
        """关闭所有 PostgreSQL 连接，并清空运行时引用。"""
        await self._stack.aclose()
        self.checkpointer = None
        self.store = None
        self.graph = None

    def require_graph(self) -> Any:
        """返回已启动的图；服务尚未启动时立即抛出明确错误。"""
        if self.graph is None:
            raise RuntimeError("PostgreSQL graph runtime has not started")
        return self.graph


runtime = PostgresGraphRuntime()
