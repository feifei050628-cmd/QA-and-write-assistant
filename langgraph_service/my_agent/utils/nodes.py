"""主图节点实现和主图与写作子图之间的适配逻辑。

可以把本模块看成一个“业务节点层”：

1. ``intent_router`` 根据显式开关或模型判断请求类型；
2. knowledge/chat/writing 节点分别处理三种业务；
3. ``knowledge_guard`` 在知识库结果不足时切换搜索兜底；
4. ``finalize`` 把内部状态整理成稳定的 answer/sources/intent。

节点返回的是状态增量，不是完整状态。消息历史由 ``add_messages`` 合并，
短期摘要由 ``SummarizationNode`` 维护，避免每个节点各自裁剪历史。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from langmem.short_term import SummarizationNode
import json
from config.settings import Settings
from my_agent.utils.state import GraphState
from my_agent.utils.tools import call_mcp_tool, web_search


# 用于正常回答的大模型。
def get_llm() -> ChatOpenAI:
    """创建统一的 LLM 实例，供图中各节点复用。"""
    return ChatOpenAI(
        model=Settings.MODEL,
        api_key=Settings.API_KEY,
        base_url=Settings.API_BASE_URL,
        temperature=Settings.TEMPERATURE,
    )


# 用于短期记忆摘要的模型。
# 这里绑定更小的 max_tokens，避免摘要本身过长。
summarization_model = get_llm().bind(max_tokens=512)

# 官方风格的短期记忆摘要节点：
# - 当历史消息过长时，自动把旧消息压缩成 running summary
# - 生成 `summarized_messages` 给后续节点使用
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=4096,
    max_tokens_before_summary=2048,
    max_summary_tokens=512,
)


def get_latest_user_query(state: GraphState) -> str:
    """从 messages 中提取最近一条用户消息，作为当前轮的 query。

    不能直接取 messages 的最后一项，因为历史中可能刚追加了 AIMessage、
    SystemMessage 或写作子图的交互消息；倒序查找 human 消息才能稳定定位本轮
    用户输入。
    """
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(item) for item in content)
            return str(content)
    return ""


def get_model_messages(state: GraphState) -> list:
    """
    获取供节点调用模型时使用的消息。

    优先使用 `summarized_messages`：
    - 这是一份"摘要后的短期记忆 + 最近原始消息"组合
    - 更适合多轮长对话

    如果摘要节点暂时没有提供输出，则退回原始 messages。
    """
    summarized_messages = state.get("summarized_messages")
    print("summarized_messages:", summarized_messages)
    if summarized_messages:
        return list(summarized_messages)
    return list(state.get("messages", []))


def is_writing_request(query: str) -> bool:
    """快速识别明确的写作请求，避免普通聊天额外调用分类模型。"""
    normalized_query = query.strip().lower()
    writing_markers = (
        "帮我写", "请写", "写一", "写个", "写篇", "写封", "写段",
        "生成", "创作", "撰写", "改写", "润色", "翻译", "编写",
        "写代码", "写个程序", "写一个程序", "write ", "translate ",
    )
    return any(marker in normalized_query for marker in writing_markers)


async def intent_router(state: GraphState) -> GraphState:
    """主智能体：识别当前请求应路由到知识库、写作还是闲聊。"""
    # 如果前端显式指定了 knowledge_bool，直接按用户意图路由。
    # 显式开关优先级高于 LLM 分类，保证用户关闭知识库时不会发生隐藏检索。
    knowledge_bool = state.get("knowledge_bool")
    if knowledge_bool is True:
        return {"intent": "knowledge"}
    if knowledge_bool is False:
        # 用户关闭知识库时，普通聊天不应先等待一次分类模型调用。
        # 明确的写作指令仍进入写作工作流，其余请求直接交给聊天节点。
        query = get_latest_user_query(state)
        return {"intent": "writing" if is_writing_request(query) else "chat"}

    # knowledge_bool 未设置，走原有 LLM 意图识别
    query = get_latest_user_query(state)
    llm = get_llm()
    prompt = f"""将用户问题分类为以下三个类别之一：

            knowledge：知识查询、事实问题、需要外部资料。
            writing：写作、生成、改写、润色、翻译、代码编写。
            chat：闲聊、观点交流、简单建议。

            规则：
            - 明确要求写/生成/翻译/润色 -> writing
            - 知识查询/事实问题 -> knowledge
            - 其他 -> chat

            用户问题：
            {query}

            只输出：
            knowledge / writing / chat"""
    result = str((await llm.ainvoke(prompt)).content).strip().lower()
    if "writing" in result or "写作" in result:
        intent = "writing"
    elif "knowledge" in result or "知识" in result or "检索" in result:
        intent = "knowledge"
    else:
        intent = "chat"
    return {"intent": intent}


def _extract_sources(raw_sources: Any) -> list[str]:
    """递归提取 sources 列表中的文本内容。"""
    if not raw_sources:
        return []
    if isinstance(raw_sources, str):
        try:
            parsed = json.loads(raw_sources)
            return _extract_sources(parsed)
        except (json.JSONDecodeError, TypeError):
            return [raw_sources]
    if isinstance(raw_sources, dict):
        text = raw_sources.get("text", "")
        if text:
            try:
                inner = json.loads(text)
                return _extract_sources(inner)
            except (json.JSONDecodeError, TypeError):
                return [text]
        return []
    if isinstance(raw_sources, list):
        result = []
        for item in raw_sources:
            result.extend(_extract_sources(item))
        return result
    return [str(raw_sources)]


def _parse_rag_response(response: dict) -> tuple[str, list[str]]:
    """解析 RAG 响应，提取 answer 和 sources。"""
    answer = str(response.get("answer", ""))
    raw_sources = response.get("sources", [])

    # 尝试从 sources 中提取嵌套的 answer
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if isinstance(item, str) and (item.startswith("[") or item.startswith("{")):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict) and parsed.get("answer"):
                        answer = parsed["answer"]
                except (json.JSONDecodeError, TypeError):
                    pass

    formatted_sources = _extract_sources(raw_sources)
    return answer, formatted_sources or (raw_sources if isinstance(raw_sources, list) else [])


async def knowledge_agent(state: GraphState) -> GraphState:
    """知识库子智能体：通过 MCP 调用现有 RAG。"""
    query = get_latest_user_query(state)
    tool_args: dict = {
        "query": query,
    }
    # knowledge_agent 不直接访问 RAG HTTP；MCP 是 Agent 与检索系统之间的
    # 边界，返回结果在这里解析为统一的 answer/sources。
    response = await call_mcp_tool("query_rag", tool_args)
    print("knowledge_agent response:", response)

    if isinstance(response, dict):
        answer, sources = _parse_rag_response(response)
        # 将sources存储到AIMessage的元数据中，实现消息级别的sources关联
        ai_message = AIMessage(
            content=answer,
            additional_kwargs={"sources": sources}
        )
        return {
            "answer": answer,
            "messages": [ai_message],
        }

    answer = str(response)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def knowledge_guard(state: GraphState) -> Literal["fallback_search", "finalize"]:
    """知识库结果守卫：知识库答不好时，切换到搜索兜底。"""
    answer = (state.get("answer") or "").strip()
    if (not answer) or ("查询失败" in answer) or ("不知道" in answer):
        return "fallback_search"
    return "finalize"


async def fallback_search_agent(state: GraphState) -> GraphState:
    """搜索兜底节点。"""
    query = get_latest_user_query(state)
    note = await web_search.ainvoke(query)
    return {
        "answer": note,
        "sources": [],
        "search_notes": [note],
        "messages": [AIMessage(content=note)],
    }


async def chat_agent(state: GraphState) -> GraphState:
    """
    普通聊天节点。

    现在不再手工 trim 历史消息，
    而是直接使用 `SummarizationNode` 产出的 `summarized_messages`。

    即：
    - 历史完整消息由 `messages + checkpointer` 自动维护
    - 长对话压缩由 `SummarizationNode` 自动维护
    - 当前节点只消费摘要结果并返回新的 AI 消息
    """
    
    llm = get_llm()
    prompt_messages = [
        SystemMessage(content="你是普通对话子智能体，直接自然回答用户。"),
        *get_model_messages(state),
    ]
    print(f"[DEBUG] chat_agent: {prompt_messages}")
    answer_parts: list[str] = []
    async for chunk in llm.astream(prompt_messages):
        if chunk.content:
            answer_parts.append(str(chunk.content))
    answer = "".join(answer_parts)
    return {
        "answer": answer,
        "sources": [],
        "messages": [AIMessage(content=answer)],
    }


def finalize(state: GraphState) -> GraphState:
    """收尾节点：返回业务字段。

    下游节点可能把 sources 放在状态字段，也可能放在 AIMessage 的
    ``additional_kwargs`` 中。这里统一从最终 AI 消息恢复来源，确保知识库
    引用不会因为经过 fallback 或子图包装而丢失。
    """
    # 从最后一条AI消息中提取sources（统一从AIMessage中获取）
    sources = []
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict):
            sources = msg.additional_kwargs.get("sources", []) or []
            break
        elif isinstance(msg, dict):
            additional_kwargs = msg.get("additional_kwargs", {})
            if isinstance(additional_kwargs, dict):
                sources = additional_kwargs.get("sources", []) or []
                break
    
    return {
        "answer": state.get("answer", ""),
        "sources": sources,
        "intent": state.get("intent", "chat"),
    }


# ==============================
# 写作子图包装函数
# ==============================
# 由于父图 (GraphState) 和子图 (WritingState) 的状态结构不同，
# 通过包装函数明确控制父图和子图之间的输入输出边界。


async def call_writing_subgraph(state: GraphState) -> dict:
    """
    写作子图包装函数：调用写作子工作流。

    状态转换说明：
    - 父图 → 子图：将 GraphState 中的字段映射为 WritingState 需要的字段
      - query (用户消息) → query

    - 子图 → 父图：将 WritingState 的输出映射回 GraphState
      - answer → answer
      - sources → sources
      - messages → messages

    Args:
        state: 父图状态 (GraphState)

    Returns:
        更新父图状态的字典
    """
    from my_agent.writing_subgraph import writing_subgraph

    # 父图只把当前用户 query 交给子图；子图内部的 requirements/outline/draft
    # 等中间字段不写入父图，防止两个状态模型互相污染。
    # 从父图状态中提取用户消息作为 query
    query = get_latest_user_query(state)
    print(f"[writing_workflow] 调用写作子图 | query={query[:50]}...")

    # 构造子图输入状态
    subgraph_input = {
        "query": query,
    }

    # 调用子图
    subgraph_output = await writing_subgraph.ainvoke(subgraph_input)

    # 将子图输出映射回父图状态
    print(f"[writing_workflow] 子图执行完成 | answer长度={len(subgraph_output.get('answer', ''))}")

    return {
        "answer": subgraph_output.get("answer", ""),
        "sources": subgraph_output.get("sources", []),
        "messages": subgraph_output.get("messages", []),
    }
