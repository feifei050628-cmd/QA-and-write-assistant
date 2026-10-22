"""完全自定义的 LangGraph 业务 API。

这个模块不依赖 ``langgraph_sdk``，也不模拟 LangGraph Server 的 threads/runs
协议。它只暴露本项目真正需要的业务接口：聊天、流式聊天、恢复中断、历史、
清空和取消。

短期记忆和长期记忆仍然由 ``app/services/graph_runtime.py`` 中编译好的图负责：
``AsyncPostgresSaver`` 保存 thread checkpoint，``AsyncPostgresStore`` 保存
跨会话的用户记忆。本模块只负责 HTTP 输入输出和任务管理。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.schemas.chat import CancelRequest, ChatRequest, ResumeRequest
from app.services.graph_runtime import runtime

router = APIRouter(prefix="/api", tags=["custom-langgraph"])


_tasks: dict[tuple[str, str], asyncio.Task[Any]] = {}

# 只有这些节点的模型输出需要展示给用户。
# summarize、intent_router、knowledge_guard 等节点属于内部编排节点，
# 如果把它们的 messages 事件也转发到前端，就可能出现内部答案和最终答案
# 同时显示，或者同一段文本被父图、子图各转发一次的问题。
_DISPLAY_NODES = {
    "format_output",
    "chat_agent",
    "knowledge_agent",
    "fallback_search",
}

_WRITING_NODES = {
    "understand",
    "ask_clarification",
    "research",
    "outline",
    "draft",
    "review",
    "human_review",
    "revise",
    "format_output",
}

_NODE_LABELS = {
    "understand": "正在理解需求...",
    "ask_clarification": "正在准备澄清问题...",
    "research": "正在搜索资料...",
    "outline": "正在生成大纲...",
    "draft": "正在撰写文章...",
    "review": "正在审核内容...",
    "human_review": "等待你审阅草稿...",
    "revise": "正在修订文章...",
    "format_output": "正在整理最终文章...",
}

# 这是运行时注入给模型的内部上下文，不属于用户可见答案。
# 这里保留前缀判断作为第二层保护：某些 LangGraph/LangChain 版本在
# subgraphs=True 时不会把消息节点元数据完整带到 messages 事件中，单纯依赖
# langgraph_node 过滤仍可能把 SystemMessage 泄漏给上游 API。
_INTERNAL_MEMORY_PREFIX = "以下是该用户的长期记忆，仅在与当前问题相关时参考："


def _thread_id(user_id: str, conversation_id: str | None) -> str:
    """将业务身份稳定映射为 PostgreSQL checkpointer 使用的 thread_id。

    前端使用的是可读的业务会话 ID，LangGraph checkpointer 使用的是 thread
    ID。用 UUID5 做确定性映射可以在服务重启后得到同一个 thread，同时把用户
    ID 和会话 ID都纳入命名空间，避免不同用户的同名会话互相串线。
    """

    conversation = conversation_id or "default"
    return str(uuid5(NAMESPACE_URL, f"custom-langgraph:{user_id}:{conversation}"))


def _config(user_id: str, thread_id: str) -> dict[str, Any]:
    """构造图配置。

    ``thread_id`` 供 checkpointer 恢复当前会话的短期状态；``user_id`` 供
    长期记忆节点选择 namespace。两者必须同时传入，否则短期会话和长期
    用户记忆可能落到错误的隔离边界。
    """

    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "langgraph_user_id": user_id,
        }
    }


def _jsonable(value: Any) -> Any:
    """将 LangChain 消息、字典和模型对象转换成 JSON 基础类型。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "content") and hasattr(value, "type"):
        return {"type": value.type, "content": _jsonable(value.content)}
    return str(value)


def _normalize_interrupt(value: Any) -> list[dict[str, Any]]:
    """把 LangGraph 的 interrupt 结果统一成前端约定的数组结构。

    ``interrupt`` 在不同 LangGraph 版本中可能是 tuple、list、单个对象，
    或者直接是 ``{"value": ...}``。前端为了兼容旧协议，统一从
    ``data.content[0].value`` 读取暂停原因，所以服务端必须在这里完成一次
    规范化。这个转换只影响 HTTP/SSE 输出，不会修改 checkpoint 中保存的
    原始中断状态。
    """
    if value is None:
        return []
    raw_items = value if isinstance(value, (list, tuple)) else [value]
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        # LangGraph 的 Interrupt 通常有 value 属性；字典形态则直接读取
        # value。若没有 value，保留整个对象，避免丢失调试信息。
        if isinstance(item, dict) and "value" in item:
            item_value = item["value"]
        else:
            item_value = getattr(item, "value", item)
        item_value = _jsonable(item_value)
        # 审阅说明由前端的 review panel 展示，不属于 AI 消息正文。
        # 删除 human_review 的 message 字段，可以阻止旧版本节点或旧 checkpoint
        # 把“请审阅草稿内容……”误传到聊天气泡；clarification 仍需保留 message
        # 作为澄清问题缺少 question 字段时的兼容兜底。
        if isinstance(item_value, dict) and item_value.get("type") == "human_review":
            item_value = {key: item for key, item in item_value.items() if key != "message"}
        normalized.append({"value": item_value})
    return normalized


def _answer(result: Any) -> dict[str, Any]:
    """从图结果中提取业务响应，同时保留 interrupt 信息。"""

    value = getattr(result, "value", result)
    interrupts = _jsonable(getattr(result, "interrupts", ()))
    state = value if isinstance(value, dict) else {"answer": value}
    messages = state.get("messages", []) or []
    answer = str(state.get("answer") or "")
    sources = state.get("sources", []) or []
    if not answer:
        for message in reversed(messages):
            message_type = message.get("type") if isinstance(message, dict) else getattr(message, "type", "")
            if message_type in {"ai", "assistant"}:
                content = message.get("content", "") if isinstance(message, dict) else message.content
                answer = content if isinstance(content, str) else str(content)
                if isinstance(message, dict):
                    sources = message.get("additional_kwargs", {}).get("sources", sources)
                break
    return {
        "answer": answer,
        "sources": _jsonable(sources),
        "intent": state.get("intent", ""),
        "conversation_id": state.get("conversation_id"),
        "is_interrupt": bool(interrupts),
        "interrupts": interrupts,
    }


def _message_and_metadata(data: Any) -> tuple[Any, dict[str, Any]]:
    """拆解 LangGraph ``messages`` 事件中的消息和元数据。

    LangGraph v2 的 messages 事件通常是 ``[message_chunk, metadata]``，
    但不同版本或不同 stream 配置下也可能只有消息对象。因此这里不依赖
    固定的列表形态，统一返回 ``(消息, 元数据)``，供流式去重逻辑使用。
    """

    if isinstance(data, (list, tuple)):
        message = data[0] if data else None
        metadata = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
        return message, metadata
    return data, {}


def _message_text(message: Any) -> str:
    """提取消息块中的文本，兼容字符串、字典和 LangChain 消息对象。"""

    if message is None:
        return ""
    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多模态消息的 content 可能是 [{"type": "text", "text": "..."}]。
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _is_internal_message(message: Any, text: str) -> bool:
    """判断消息是否属于只供模型使用的内部上下文。

    长期记忆以 ``SystemMessage`` 注入图状态。即便事件没有 metadata，
    也可以通过消息类型和固定前缀识别它。返回 True 时，该消息不会生成
    token SSE，但它仍然保留在图状态中，不影响模型使用长期记忆。
    """

    message_type = message.get("type") if isinstance(message, dict) else getattr(message, "type", "")
    return message_type == "system" or text.startswith(_INTERNAL_MEMORY_PREFIX)


def _is_message_chunk(message: Any) -> bool:
    """只识别真正的流式消息块，避免转发节点完成后的完整消息。"""

    message_type = message.get("type", "") if isinstance(message, dict) else getattr(message, "type", "")
    class_name = type(message).__name__
    return str(message_type).endswith("MessageChunk") or class_name.endswith("MessageChunk")


def _public_history(messages: Any) -> list[Any]:
    """只保留前端需要展示的历史消息。

    checkpoint 中除了 human/ai 消息，还可能包含长期记忆 SystemMessage、
    摘要消息以及其他内部消息。历史接口没有必要把这些内容序列化并通过
    HTTP 传输，提前过滤可以降低响应体大小和前端解析时间。
    """

    if not isinstance(messages, list):
        return []
    result: list[Any] = []
    for message in messages:
        text = _message_text(message)
        if _is_internal_message(message, text):
            continue
        message_type = message.get("type") if isinstance(message, dict) else getattr(message, "type", "")
        if message_type in {"human", "user", "ai", "assistant"}:
            result.append(message)
    return result


def _writing_node_name(part: dict[str, Any], data: Any) -> str:
    """从 v2 事件中推断写作子图当前节点名称。

    开启 ``subgraphs=True`` 后，LangGraph 不同版本对 ``ns`` 的表示略有
    差异，可能是 ``writing_workflow:uuid``，也可能把节点名放在 updates
    的字典键中。因此这里同时检查 data 键和 ns 字符串，避免绑定某一个
    具体版本的内部事件结构。
    """

    field_to_node = {
        "draft": "draft",
        "outline": "outline",
        "formatted_content": "format_output",
        "review_feedback": "review",
        "research_notes": "research",
    }

    def find_in_mapping(mapping: Any) -> str:
        if not isinstance(mapping, dict):
            return ""
        # 使用包装函数调用子图时，子图的最终结果会作为
        # {"writing_workflow": {"answer": ...}} 回到父图。这里只把
        # writing_workflow 下的 answer 识别为最终文章，避免把普通
        # finalize.answer 误转成第二份 content 事件。
        writing_result = mapping.get("writing_workflow")
        if isinstance(writing_result, dict) and (
            "answer" in writing_result or "formatted_content" in writing_result
        ):
            return "format_output"
        # 子图更新有时是 {"writing_workflow": {"draft": ...}}，
        # 有时是 {"draft": ...}。递归检查可以同时兼容两类结构。
        for key, nested in mapping.items():
            if key in _WRITING_NODES:
                return key
            if key in field_to_node:
                return field_to_node[key]
            nested_node = find_in_mapping(nested)
            if nested_node:
                return nested_node
        return ""

    node_name = find_in_mapping(data)
    if node_name:
        return node_name

    namespaces = part.get("ns", [])
    if isinstance(namespaces, (list, tuple)):
        for namespace in reversed(namespaces):
            text = str(namespace)
            for node_name in _WRITING_NODES:
                if node_name in text:
                    return node_name
    return ""


def _update_payload(data: Any, node_name: str) -> dict[str, Any]:
    """提取某个写作节点的状态更新。"""

    if not isinstance(data, dict):
        return {}
    # 包装调用模式下，子图结果由父图节点一次性返回：
    # {"writing_workflow": {"answer": ..., "messages": [...]}}。
    if node_name == "format_output":
        writing_result = data.get("writing_workflow")
        if isinstance(writing_result, dict) and (
            "answer" in writing_result or "formatted_content" in writing_result
        ):
            return writing_result
    if node_name in data and isinstance(data[node_name], dict):
        return data[node_name]

    # updates 可能把子图名称放在外层，例如：
    # {"writing_workflow": {"draft": {"draft": "..."}}}。
    # 逐层查找真正的节点状态，避免 payload 只拿到外层包装对象。
    for nested in data.values():
        payload = _update_payload(nested, node_name)
        if payload:
            return payload
    return data


def _answer_update_event(data: Any) -> tuple[str, dict[str, Any]] | None:
    """把一次性返回答案的节点更新转换成前端 content 事件。

    knowledge_agent 和 fallback_search_agent 使用 ``ainvoke``，更新中携带
    完整的 ``answer``，不会产生 ``AIMessageChunk``。普通 chat_agent 则通过
    messages 事件发送 token，因此不能把所有节点的 answer 都再次转发。
    """

    if not isinstance(data, dict):
        return None
    for node_name in ("knowledge_agent", "fallback_search"):
        payload = data.get(node_name)
        if not isinstance(payload, dict):
            continue
        text = str(payload.get("answer") or "")
        sources = payload.get("sources", []) or []
        if not sources:
            for message in payload.get("messages", []) or []:
                if not isinstance(message, dict):
                    continue
                additional_kwargs = message.get("additional_kwargs", {})
                if isinstance(additional_kwargs, dict):
                    sources = additional_kwargs.get("sources", []) or []
                    if sources:
                        break
        if text:
            return "content", {
                "content": text,
                "sources": _jsonable(sources),
                "is_display": True,
            }
    return None


def _writing_events(
    part: dict[str, Any],
    started_nodes: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    """把 LangGraph updates 转换成前端使用的写作状态和最终内容事件。

    写作子图中的 ``outline``、``draft``、``review`` 等节点大多使用
    ``ainvoke``，不会产生普通 token 流；它们的结果会出现在 updates 中。
    草稿正文在 interrupt 中发送给前端审阅，只有 format_output 的最终结果
    需要在这里转换为 content 事件。
    """

    data = part.get("data")
    node_name = _writing_node_name(part, data)
    if not node_name:
        return []

    result: list[tuple[str, dict[str, Any]]] = []
    if node_name not in started_nodes:
        started_nodes.add(node_name)
        result.append(("node_start", {
            "node": node_name,
            "label": _NODE_LABELS.get(node_name, f"正在执行 {node_name}..."),
            "is_display": node_name in {"draft", "format_output"},
        }))

    payload = _update_payload(data, node_name)
    # 子图内部的 format_output 只保留为状态更新；最终文章统一从父图
    # 当前拿到的是否是“父图 writing_workflow 包装节点返回的最终结果
    if node_name == "format_output" and isinstance(data, dict) and "writing_workflow" in data:
        text = str(payload.get("answer") or payload.get("formatted_content") or "")
        if text:
            result.append(("content", {"content": text, "is_display": True}))

    return result


async def _run(
    user_id: str,
    conversation_id: str,
    input_value: Any,
    run_id: str,
) -> Any:
    """登记并执行一个图任务，供取消接口调用 ``Task.cancel``。"""

    # 任务登记只存在当前 Uvicorn worker 的内存中，目的是让 cancel 接口能在
    # 同一个 worker 内调用 Task.cancel；真正的对话状态仍由 PostgreSQL 持久化。
    thread_id = _thread_id(user_id, conversation_id)
    task = asyncio.current_task()
    if task is not None:
        _tasks[(thread_id, run_id)] = task
    try:
        return await runtime.require_graph().ainvoke(input_value, config=_config(user_id, thread_id))
    finally:
        _tasks.pop((thread_id, run_id), None)


@router.get("/health")
async def custom_health() -> dict[str, str]:
    """自定义业务 API 健康检查。"""

    return {"status": "ok" if runtime.graph is not None else "starting"}


@router.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """非流式聊天接口。"""

    conversation_id = request.conversation_id or str(uuid.uuid4())
    # 请求只携带本轮新增消息，历史由 thread_id 对应的 checkpointer 自动恢复。
    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
    }
    if request.knowledge_bool is not None:
        input_state["knowledge_bool"] = request.knowledge_bool
    run_id = str(uuid.uuid4())
    try:
        result = await _run(request.user_id, conversation_id, input_state, run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    response = _answer(result)
    response.update({"conversation_id": conversation_id, "run_id": run_id})
    return response


async def _stream(request: ChatRequest, run_id: str) -> AsyncIterator[str]:
    """将图的 messages/updates 转换成简单、稳定的自定义 SSE 事件。"""

    # 流式和非流式使用同一套 thread 映射，保证两种接口可以交替访问同一会话。
    thread_id = _thread_id(request.user_id, request.conversation_id)
    task = asyncio.current_task()
    if task is not None:
        _tasks[(thread_id, run_id)] = task
    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
    }
    if request.knowledge_bool is not None:
        input_state["knowledge_bool"] = request.knowledge_bool
    started_nodes: set[str] = set()
    try:
        # 前端按 conversation_id 事件保存新会话；start 事件保留给需要
        # 运行元数据的客户端，两个事件都由 LangGraph 服务统一生成。
        yield _event("conversation_id", {"content": request.conversation_id})
        yield _event("start", {"run_id": run_id, "conversation_id": request.conversation_id})
        async for part in runtime.require_graph().astream(
            input_state,
            config=_config(request.user_id, thread_id),
            stream_mode=["messages", "updates"],
            subgraphs=True,
            version="v2",
        ):
            if not isinstance(part, dict):
                continue
            event_type = part.get("type")
            data = part.get("data")
            if event_type == "messages":
                message, metadata = _message_and_metadata(data)
                node_name = str(metadata.get("langgraph_node", ""))
                text = _message_text(message)
                if _is_internal_message(message, text):
                    continue
                if not _is_message_chunk(message):
                    continue
                # 有节点元数据时严格按展示节点过滤；没有元数据时保留兼容性。
                #
                # 这里不能把所有 AIMessage 都转发出去，因为 summarize、路由、
                # 守卫节点也可能产生消息，它们是 Agent 内部推理过程，不是最终
                # 答案。长期记忆 SystemMessage 也在更早的分支被过滤。
                if node_name and node_name not in _DISPLAY_NODES:
                    continue
                if text:
                    yield _event("content", {"content": text, "is_display": True})
            elif event_type == "updates":
                answer_event = _answer_update_event(data)
                if answer_event:
                    yield _event(*answer_event)
                # updates 承载的是“节点完成后的状态变化”，写作节点的状态和
                # 最终文章主要从这里出来。先转换成前端理解的业务事件，
                # 再保留一个通用 update 事件，方便调试和未来扩展字段。
                for writing_event_type, writing_event in _writing_events(part, started_nodes):
                    yield _event(writing_event_type, writing_event)
                yield _event("update", _jsonable(data))
                if isinstance(data, dict) and data.get("__interrupt__"):
                    # interrupt 必须和 node update 分开通知前端，否则前端会把
                    # “等待用户操作”误认为普通更新，无法显示澄清/审阅界面。
                    yield _event("interrupt", _normalize_interrupt(data["__interrupt__"]))
        yield _event("complete", {"finished": True, "conversation_id": request.conversation_id, "run_id": run_id})
    except asyncio.CancelledError:
        yield _event("cancelled", {"run_id": run_id})
        raise
    except Exception as exc:
        yield _event("error", {"message": str(exc)})
    finally:
        _tasks.pop((thread_id, run_id), None)


def _event(event_type: str, data: Any) -> str:
    """生成自定义 SSE；客户端只需要解析 data 中的 type 字段。"""

    payload = {"type": event_type, **(data if isinstance(data, dict) else {"content": data})}
    return f"data: {json.dumps(_jsonable(payload), ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式聊天接口，不使用 LangGraph SDK 的 v2 协议。"""

    conversation_id = request.conversation_id or str(uuid.uuid4())
    request = request.model_copy(update={"conversation_id": conversation_id})
    run_id = str(uuid.uuid4())
    return StreamingResponse(
        _stream(request, run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Run-Id": run_id},
    )


@router.get("/conversations/{conversation_id}/history")
async def history(conversation_id: str, user_id: str) -> dict[str, Any]:
    """读取当前用户当前会话的短期消息历史。"""

    thread_id = _thread_id(user_id, conversation_id)
    try:
        snapshot = await runtime.require_graph().aget_state(_config(user_id, thread_id))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"conversation_id": conversation_id, "messages": _jsonable(_public_history(snapshot.values.get("messages", [])))}


@router.delete("/conversations/{conversation_id}")
async def clear(conversation_id: str, user_id: str) -> dict[str, str]:
    """删除当前会话的短期 checkpoint，长期记忆按设计保留。"""

    checkpointer = runtime.checkpointer
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="runtime is not ready")
    await checkpointer.adelete_thread(_thread_id(user_id, conversation_id))
    return {"status": "success", "message": "conversation cleared"}


@router.post("/chat/cancel")
async def cancel(request: CancelRequest) -> dict[str, Any]:
    """取消当前 worker 中的 asyncio 任务；多 worker 需改为共享任务队列。"""

    thread_id = _thread_id(request.user_id, request.conversation_id)
    keys = [(thread_id, request.run_id)] if request.run_id else [key for key in _tasks if key[0] == thread_id]
    cancelled = 0
    for key in keys:
        task = _tasks.get(key)
        if task is not None:
            task.cancel()
            cancelled += 1
    return {"status": "cancelled", "cancelled": cancelled}


@router.post("/chat/resume")
async def resume(request: ResumeRequest) -> StreamingResponse:
    """恢复被 interrupt 暂停的图，并继续返回自定义 SSE。"""

    thread_id = _thread_id(request.user_id, request.conversation_id)
    # interrupt 所在节点约定了两类恢复载荷：澄清节点读取 answer，人工审阅
    # 节点读取 action/content。这里把 HTTP 请求模型转换成图节点约定的结构。
    resume_data: dict[str, Any]
    if request.action == "clarify":
        resume_data = {"answer": request.answer or request.content or ""}
    else:
        resume_data = {"action": request.action}
        if request.action == "edit" and request.content is not None:
            resume_data["content"] = request.content
    run_id = str(uuid.uuid4())

    async def events() -> AsyncIterator[str]:
        task = asyncio.current_task()
        if task is not None:
            _tasks[(thread_id, run_id)] = task
        started_nodes: set[str] = set()
        try:
            async for part in runtime.require_graph().astream(
                Command(resume=resume_data),
                config=_config(request.user_id, thread_id),
                stream_mode=["messages", "updates"],
                subgraphs=True,
                version="v2",
            ):
                if isinstance(part, dict):
                    event_type = str(part.get("type", "update"))
                    data = part.get("data")
                    if event_type == "messages":
                        message, metadata = _message_and_metadata(data)
                        node_name = str(metadata.get("langgraph_node", ""))
                        text = _message_text(message)
                        if _is_internal_message(message, text):
                            continue
                        if not _is_message_chunk(message):
                            continue
                        if node_name and node_name not in _DISPLAY_NODES:
                            continue
                        if text:
                            yield _event("content", {"content": text, "is_display": True})
                    elif event_type == "updates":
                        answer_event = _answer_update_event(data)
                        if answer_event:
                            yield _event(*answer_event)
                        for writing_event_type, writing_event in _writing_events(part, started_nodes):
                            yield _event(writing_event_type, writing_event)
                        yield _event("update", _jsonable(data))
                        if isinstance(data, dict) and data.get("__interrupt__"):
                            yield _event("interrupt", _normalize_interrupt(data["__interrupt__"]))
                    else:
                        yield _event(event_type, _jsonable(data))
            yield _event("complete", {"finished": True})
        finally:
            _tasks.pop((thread_id, run_id), None)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Run-Id": run_id})
