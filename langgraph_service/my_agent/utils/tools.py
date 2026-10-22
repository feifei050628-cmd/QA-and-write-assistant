"""主图使用的外部工具。

这里把外部系统访问集中起来：MCP 负责调用 RAG，Tavily 负责搜索兜底，
langmem 工具负责长期记忆管理。节点只需要调用这些工具，不需要关心具体的
HTTP/MCP 协议细节。
"""

import json
import os
from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langmem import create_manage_memory_tool, create_search_memory_tool
from mcp import ClientSession

from config.settings import Settings

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None


from mcp.client.streamable_http import streamable_http_client


async def _call_via_langchain(tool_name: str, arguments: dict[str, Any]) -> Any:
    """通过 LangChain MCP adapter 调用工具。

    adapter 能把远端 MCP 工具包装成 LangChain Tool，适合和其他 LangChain
    组件统一使用。函数只负责一次调用，不持有跨请求的客户端对象。
    """
    if MultiServerMCPClient is None:
        raise ImportError("langchain-mcp-adapters is not installed")

    client = MultiServerMCPClient(
        {
            "rag": {
                "transport": "streamable_http",
                "url": Settings.MCP_SERVICE_URL,
            }
        }
    )
    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools}
    if tool_name not in tool_map:
        raise ValueError(f"MCP tool not found: {tool_name}")
    return await tool_map[tool_name].astream(arguments)


async def _call_via_mcp(tool_name: str, arguments: dict[str, Any]) -> Any:
    """直接使用 MCP streamable-http 协议调用工具。

    这是 adapter 不可用或调用失败时的兼容路径。MCP 返回值可能是结构化内容，
    也可能是文本 JSON，因此这里统一尽量还原为 Python 对象。
    """
    async with streamable_http_client(Settings.MCP_SERVICE_URL) as (read, write, *_):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if hasattr(result, "structuredContent") and result.structuredContent:
                return result.structuredContent
            if hasattr(result, "content") and result.content:
                first = result.content[0]
                text = getattr(first, "text", str(first))
                try:
                    return json.loads(text)
                except Exception:
                    return text
            return result


async def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """调用 RAG MCP 工具，并在两种 MCP 客户端实现之间自动降级。"""
    try:
        return await _call_via_langchain(tool_name, arguments)
    except Exception:
        return await _call_via_mcp(tool_name, arguments)


# ==============================
# Tavily 搜索兜底（懒加载）
# ==============================
# Tavily 是可选能力：没有配置 TAVILY_API_KEY 时，知识库兜底搜索不可用，
# 但主图、普通聊天和知识库问答必须仍然能够正常启动和使用。
# 因此这里不在模块导入时构造 TavilySearch（它会在构造期强制校验 API Key，
# 一旦缺失会让整个服务启动失败），而是等到真正需要搜索时再创建。
_tavily_search: TavilySearch | None = None


def _get_tavily_search() -> TavilySearch:
    """按需创建 TavilySearch 单例；缺少 Key 时给出明确错误。"""
    global _tavily_search
    if _tavily_search is None:
        api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "未配置 TAVILY_API_KEY，知识库兜底搜索不可用；"
                "如需该功能请在项目根目录 .env 中填写 Tavily API Key。"
            )
        _tavily_search = TavilySearch(
            max_results=5,
            topic="general",
            tavily_api_key=api_key,
        )
    return _tavily_search


@tool
async def web_search(query: str) -> str:
    """搜索互联网上的公开信息，用于知识库不足时补充事实。"""
    try:
        # Tavily 返回的是搜索结果集合，而 fallback 节点只需要一段可以直接
        # 交给模型或前端的文本，所以在工具边界完成格式化。
        results = await _get_tavily_search().ainvoke({"query": query})
        # 格式化搜索结果
        if isinstance(results, dict):
            # TavilySearch 返回格式
            search_results = results.get("results", [])
            if search_results:
                formatted = []
                for i, result in enumerate(search_results[:5], 1):
                    title = result.get("title", "")
                    url = result.get("url", "")
                    content = result.get("content", "")
                    formatted.append(f"{i}. {title}\n   链接: {url}\n   摘要: {content}")
                return "\n\n".join(formatted)
            return "未找到相关搜索结果"
        elif isinstance(results, str):
            return results
        else:
            return str(results)
    except Exception as e:
        return f"搜索出错: {str(e)}"


# ==============================
# langmem 长期记忆工具
# ==============================
# 使用官方 langmem 工具实现长期记忆
# namespace 支持按用户隔离：("memories", "{user_id}")
memory_tools = [
    create_manage_memory_tool(namespace=("memories", "{user_id}")),
    create_search_memory_tool(namespace=("memories", "{user_id}")),
]
