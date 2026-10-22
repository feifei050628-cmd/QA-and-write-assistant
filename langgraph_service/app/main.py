"""方案二专用的精简 FastAPI 应用。

这里只导入自定义业务路由，不包含 LangGraph Server/SDK 的 threads/runs 协议，
方案二完全不需要这些兼容接口。

本应用只注册 ``app/api/chat.py`` 中定义的业务接口，并复用同一个 PostgreSQL
图运行时，因此短期 checkpoint、长期记忆和 interrupt 恢复逻辑保持一致。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as custom_api_router
from app.services.graph_runtime import runtime


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时初始化 PostgreSQL 图，关闭时释放连接池。

    只有 startup 成功后 runtime.graph 才会赋值，所以数据库连接、表结构和
    编译图是一个整体的启动前置条件；这样不会出现 HTTP 已在线但图尚未可用
    的半初始化状态。
    """

    # FastAPI 会在接受第一个请求前进入这里。graph_runtime 在此阶段创建
    # PostgreSQL 连接、初始化 checkpoint/store 表，并 compile 出真正执行的图；
    # 如果启动阶段失败，应用不会对外提供一个“看起来在线但没有记忆”的服务。
    await runtime.startup()
    try:
        yield
    finally:
        # 无论应用是正常关闭还是收到退出信号，都通过同一个 runtime 释放
        # checkpointer 和 store 的异步连接，避免连接池残留在 PostgreSQL 中。
        await runtime.shutdown()


app = FastAPI(
    title="LangGraph Custom FastAPI Service",
    version="1.0.0",
    description="不依赖 langgraph-sdk 的自定义 LangGraph 业务 API",
    lifespan=lifespan,
)
app.include_router(custom_api_router)
# router 内部已经包含 /api 前缀，因此这里不再重复添加前缀。
# 这样最终接口是 /api/chat、/api/chat/stream、/api/conversations/...。


@app.get("/")
async def root() -> dict[str, str]:
    """返回当前部署模式。"""

    return {"service": "langgraph_service", "deployment": "custom-fastapi"}


@app.get("/health")
async def health() -> dict[str, str]:
    """检查 FastAPI 进程和图运行时状态。"""

    # “ok”不仅表示 HTTP 进程存在，还表示 PostgreSQL 图已经编译完成；
    # 启动期间数据库尚未连接时返回 starting，便于反向代理做健康检查。
    return {"status": "ok" if runtime.graph is not None else "starting"}
