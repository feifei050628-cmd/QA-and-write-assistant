"""方案二专用启动入口。

启动命令：

    uv run --project langgraph_service python langgraph_service/run.py

该入口只加载 ``app.main:app``，不包含 LangGraph Server/SDK 兼容接口。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import uvicorn


def _force_utf8_stdio() -> None:
    """把标准输出/错误流切换为 UTF-8。

    Windows 控制台默认使用 GBK（cp936）。节点里存在调试 ``print()``，当模型
    回答包含 emoji（例如 😊）时，打印会抛 ``UnicodeEncodeError``，异常沿图执行
    传播，最终表现为一轮 SSE 直接输出 error 事件。
    ``errors="replace"`` 保证即使控制台字体不支持某些字符也不会中断业务。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()


def run_server(app: object, *, host: str, port: int) -> None:
    """使用 Psycopg 兼容的事件循环运行 Uvicorn。

    Windows 的默认 ``ProactorEventLoop`` 不能用于 Psycopg 异步连接。
    这里直接创建 ``SelectorEventLoop``，并让 Uvicorn 的 ``Server.serve``
    在这个循环中运行，从而不需要调用已经被新版 Python 标记为弃用的
    ``asyncio.set_event_loop_policy``。

    ``loop="none"`` 很重要：它告诉 Uvicorn 不要再次创建或替换事件循环，
    统一使用下面显式创建的循环。Linux/macOS 也可以使用这个实现，Selector
    循环对本服务的异步 HTTP 和 PostgreSQL 操作没有问题。
    """

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        reload=False,
        loop="none",
    )
    server = uvicorn.Server(config)
    # Windows 默认 Proactor 循环与 psycopg async 连接不兼容。显式创建
    # SelectorEventLoop 后，下面的 run_until_complete 会让 Uvicorn、FastAPI
    # lifespan 和 PostgreSQL 全部运行在同一个兼容循环中。
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


if __name__ == "__main__":
    # 直接执行 ``python langgraph_service/run.py`` 时，Python 默认把脚本所在
    # 目录加入 sys.path。这里显式补充一次，避免不同启动目录导致 ``app``、
    # ``config`` 或 ``my_agent`` 无法导入。
    service_dir = str(Path(__file__).resolve().parent)
    if service_dir not in sys.path:
        sys.path.insert(0, service_dir)

    # 延迟导入 app：先修正 sys.path 和事件循环相关配置，再加载会创建运行时
    # 对象的模块，直接执行脚本时对 Windows 路径更稳定。
    # 这里采用延迟导入：app.main 导入时会创建 runtime，先处理 sys.path 和
    # 事件循环，再导入可以避免直接执行脚本时出现模块或连接初始化顺序问题。
    from app.main import app

    run_server(
        app,
        host="0.0.0.0",
        port=int(os.getenv("LANGGRAPH_FASTAPI_PORT", "2024")),
    )
