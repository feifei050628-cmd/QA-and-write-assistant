"""LangGraph CLI 图的构建入口。

主图把通用对话编排与各个业务节点连接起来。FastAPI 自定义部署会在
``app/services/graph_runtime.py`` 重新构建一份拓扑相同、但额外带 PostgreSQL
长期记忆边界节点的图；两者共享节点实现，避免业务逻辑分叉。
"""

import asyncio
import sys

from langgraph.graph import END, START, StateGraph

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
from my_agent.utils.tools import memory_tools

if sys.platform.startswith("win"):
    # Psycopg 异步连接不支持 Windows 默认的 ProactorEventLoop。
    # 直接安装 SelectorEventLoop，避免调用新版 Python 已标记为弃用的
    # asyncio.set_event_loop_policy()。
    asyncio.set_event_loop(asyncio.SelectorEventLoop())



# ==============================
# 构建 LangGraph 图
# ==============================
# 这里采用"先摘要，再路由"的官方短期记忆模式：
# START -> summarize -> intent_router -> ...
builder = StateGraph(GraphState)
builder.add_node("summarize", summarization_node)
builder.add_node("intent_router", intent_router)
builder.add_node("knowledge_agent", knowledge_agent)
builder.add_node("fallback_search", fallback_search_agent)
builder.add_node("writing_workflow", call_writing_subgraph)  # 使用包装函数
builder.add_node("chat_agent", chat_agent)
builder.add_node("finalize", finalize)

# 图入口先做短期记忆摘要。
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "intent_router")

# 根据识别出的 intent 路由到对应子智能体。
# intent_router 只写入 intent，不直接调用下游节点；条件边负责真正选择
# knowledge、writing 或 chat 分支，因此路由逻辑和业务处理逻辑彼此隔离。
builder.add_conditional_edges(
    "intent_router",
    lambda state: state["intent"],
    {
        "knowledge": "knowledge_agent",
        "writing": "writing_workflow",
        "chat": "chat_agent",
    },
)

# 知识库问答如果结果差，就自动切换到搜索兜底。
# 该分支只会在 knowledge_agent 完成后判断，不会影响普通聊天和写作流程。
builder.add_conditional_edges(
    "knowledge_agent",
    knowledge_guard,
    {
        "fallback_search": "fallback_search",
        "finalize": "finalize",
    },
)

# 搜索兜底直接到收尾。
builder.add_edge("fallback_search", "finalize")

# 写作子工作流完成后直接到收尾。
builder.add_edge("writing_workflow", "finalize")

# 普通聊天直接进入收尾。
builder.add_edge("chat_agent", "finalize")

# finalize 之后结束整轮执行。
builder.add_edge("finalize", END)

# 编译图时传入 store，让 langmem 工具可以访问
graph = builder.compile()
