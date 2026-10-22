"""langgraph_service 的环境变量配置。

项目根目录的 ``.env`` 会在导入时加载。这里集中保存模型、数据库和 MCP
地址，业务模块只依赖 Settings，避免在多个节点里重复读取环境变量。
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """langgraph_service 配置。

    ``POSTGRES_URL`` 是 SQLAlchemy 风格地址；``LANGGRAPH_POSTGRES_DSN`` 是
    Psycopg/LangGraph 使用的 DSN。两者都指向同一个 PostgreSQL 数据库，但
    由不同组件消费，不能简单混用格式。
    """

    PROJECT_ROOT: Path = Path(__file__).parent.parent

    API_KEY: Optional[str] = os.getenv("DASHSCOPE_API_KEY")
    API_BASE_URL: Optional[str] = os.getenv("DASHSCOPE_BASE_URL")
    MODEL: str =  os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.1"))

    POSTGRES_URL: str = os.getenv(
        "POSTGRES_URL",
        "postgresql+psycopg://postgres:123456@127.0.0.1:5432/langgraph_agents",
    )
    LANGGRAPH_POSTGRES_DSN: str = os.getenv(
        "LANGGRAPH_POSTGRES_DSN",
        "postgresql://postgres:123456@127.0.0.1:5432/langgraph_agents?sslmode=disable",
    )
    MCP_SERVICE_URL: str = os.getenv("MCP_SERVICE_URL", "http://127.0.0.1:8010/mcp")
