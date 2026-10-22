"""旧 LangGraph SDK 聊天路由。

本路由把 SDK 的 ``messages/metadata``、``messages/partial`` 和 ``updates``
事件翻译成前端使用的简化 SSE。metadata 决定当前节点，partial 提供 token，
updates 提供写作节点的完整结果或 interrupt；三类事件的职责不同，不能混为
一个普通文本流。
"""

from typing import Any, List
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.routers.users import get_current_active_user
from app.schemas import (
    ChatMessage, ChatRequest, ChatResponse, ClearRequest, CommonResponse,
    ConversationCreate, ConversationListResponse, ConversationResponse, ConversationUpdate,
    ResumeRequest, User,
)
from app.services import get_langgraph_service
from app.services.conversation_repository import ConversationRepository
from app.services.langgraph_service import LangGraphService
from db import get_db
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

DISPLAY_NODES = {"format_output", "chat_agent", "knowledge_agent", "fallback_search"}
TRACK_NODES = {
    "understand", "ask_clarification", "research", "outline", "draft",
    "review", "revise", "format_output", "chat_agent", "knowledge_agent",
    "fallback_search",
}
NODE_LABELS = {
    "understand": "正在理解需求...",
    "ask_clarification": "正在准备提问...",
    "research": "正在搜索资料...",
    "outline": "正在生成大纲...",
    "draft": "正在撰写文章...",
    "review": "正在审核内容...",
    "revise": "正在修订内容...",
    "format_output": "正在格式化输出...",
    "chat_agent": "正在思考...",
    "knowledge_agent": "正在检索知识库...",
    "fallback_search": "正在搜索...",
}


def sse_event(event_type: str, *, content=None, finished: bool | None = None, **fields) -> str:
    """构造统一的精简 SSE 事件。

    前端只需解析 ``data`` 中的 JSON；中间事件不重复携带 finished=false，
    只有 complete/error 等收尾事件才明确表达结束状态。
    """
    payload = {"type": event_type}
    if content is not None:
        payload["content"] = content
    if finished is not None:
        payload["finished"] = finished
    payload.update(fields)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def normalize_interrupt_payload(value: Any) -> Any:
    """清理发给前端的人机交互载荷。

    ``human_review`` 的草稿正文位于 ``draft``，审阅说明由前端面板负责展示，
    因此不能再把同一说明放进 ``message``。这里兼容列表、``{value: ...}``
    和直接字典三种 LangGraph 返回形态，避免旧 checkpoint 中的 ``message``
    字段被误当成 AI 消息正文；澄清中断仍保留 ``message`` 作为兼容字段。
    """
    if isinstance(value, list):
        return [normalize_interrupt_payload(item) for item in value]
    if isinstance(value, dict):
        if "value" in value:
            return {**value, "value": normalize_interrupt_payload(value["value"])}
        if value.get("type") == "human_review":
            return {key: item for key, item in value.items() if key != "message"}
    return value


def normalize_sources(sources: Any) -> list[str]:
    """递归解包 RAG 来源，统一转换为前端可展示的文本列表。

    RAG 返回值在不同版本中可能是列表、``{"text": ...}``、嵌套 JSON 字符串，
    所以在 API 边界统一清洗，前端不需要理解下游格式差异。
    """
    normalized: list[str] = []

    def append_source(source: Any) -> None:
        if source is None:
            return

        if isinstance(source, list):
            for item in source:
                append_source(item)
            return

        if isinstance(source, dict):
            if "text" in source:
                append_source(source.get("text"))
                return

            append_source(source.get("sources"))
            return

        if isinstance(source, str):
            stripped_source = source.strip()
            if stripped_source.startswith(("[", "{")):
                try:
                    append_source(json.loads(stripped_source))
                    return
                except (json.JSONDecodeError, TypeError):
                    pass
            if source:
                normalized.append(source)
            return

        normalized.append(str(source))

    append_source(sources)
    return normalized


# ===== 会话管理 API =====

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    req: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """创建新会话。"""
    repo = ConversationRepository(db)
    conversation = repo.create(user_id=current_user.username, title=req.title or "新对话")
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """获取当前用户的会话列表。"""
    repo = ConversationRepository(db)
    conversations = repo.list_by_user(current_user.username)
    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in conversations
        ]
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    req: ConversationUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """更新会话标题。"""
    repo = ConversationRepository(db)
    conversation = repo.update_title(conversation_id, current_user.username, req.title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.delete("/conversations/{conversation_id}", response_model=CommonResponse)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    svc: LangGraphService = Depends(get_langgraph_service),
):
    """删除会话（同时删除 LangGraph Thread 状态）。"""
    repo = ConversationRepository(db)
    if not repo.delete(conversation_id, current_user.username):
        raise HTTPException(status_code=404, detail="会话不存在")
    # 同时删除 LangGraph Thread 状态
    try:
        await svc.clear_history(thread_id=current_user.id, conversation_id=conversation_id)
    except Exception as e:
        logger.warning(f"删除 LangGraph Thread 失败: {e}")
    return CommonResponse(status="success", message="会话已删除")


# ===== 聊天 API =====


@router.post("/chat", response_model=ChatResponse)
async def chat_query(
    req: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    svc: LangGraphService = Depends(get_langgraph_service),
    db: Session = Depends(get_db),
):
    """
    非流式聊天接口。

    改造后的行为：
    - 不再直接调用原有 RAGService 做对话
    - 而是统一转发到本地部署的 LangGraph 服务

    请求体现在只保留两个核心字段：
    - query
    - model

    用户身份仍由 FastAPI 的 JWT 鉴权负责，
    然后把稳定的 user_id 作为业务身份传给 LangGraph，
    用于隔离不同用户的记忆与状态。
    """
    # 如果没有提供 conversation_id，先在 main_service 会话表创建业务会话。
    # 这个 ID随后同时传给 LangGraph service，成为短期 checkpoint 隔离键的一部分。
    conversation_id = req.conversation_id
    if not conversation_id:
        repo = ConversationRepository(db)
        # 自动生成标题（取前20个字符）
        title = req.query[:20] + "..." if len(req.query) > 20 else req.query
        conversation = repo.create(user_id=current_user.username, title=title)
        conversation_id = conversation.id
    else:
        # 更新会话的最后活跃时间
        repo = ConversationRepository(db)
        repo.touch(conversation_id, current_user.username)

    try:
        result = await svc.chat(
            thread_id=current_user.id,
            conversation_id=conversation_id,
            query=req.query,
            knowledge_bool=req.knowledge_bool,
        )
        
        # 检测是否是 interrupt（人机交互）
        if result.get("is_interrupt"):
            return ChatResponse(
                messages=ChatMessage(
                    role="ai",
                    content=result.get("answer", ""),
                    sources=[],
                ),
                conversation_id=conversation_id,
                is_interrupt=True,
            )
        
        return ChatResponse(
            messages=ChatMessage(
                role="ai",
                content=result.get("answer", ""),
                sources=result.get("sources", []) or [],
            ),
            conversation_id=conversation_id,
            is_interrupt=False,
        )
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_query_stream(
    req: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    svc: LangGraphService = Depends(get_langgraph_service),
    db: Session = Depends(get_db),
):
    """
    流式聊天接口（真正的 token 级流式输出）。

    使用 LangGraph SDK 的 stream 方法，stream_mode=["messages", "updates"]。
    - messages 模式：token 级别的流式输出
    - updates 模式：节点级别的状态更新
    """
    # 如果没有提供 conversation_id，自动创建新会话
    conversation_id = req.conversation_id
    if not conversation_id:
        repo = ConversationRepository(db)
        title = req.query[:20] + "..." if len(req.query) > 20 else req.query
        conversation = repo.create(user_id=current_user.username, title=title)
        conversation_id = conversation.id
    else:
        repo = ConversationRepository(db)
        repo.touch(conversation_id, current_user.username)

    async def event_generator():
        # 事件生成器在真正消费下游流时才执行，因此可以感知客户端断开并及时
        # 取消下游 run，避免浏览器已经关闭后模型仍继续生成。
        try:
            # 先立即刷新一个轻量事件，避免客户端在 LangGraph 建立 run
            # 和执行前置节点期间误以为连接已失效。
            yield sse_event(
                "status",
                content="请求已接收，正在连接智能体...",
                is_display=False,
            )
            final_sources = []
            prev_content = ""
            current_node = ""
            streamed_display_nodes: set[str] = set()
            async for event in svc.chat_stream(
                thread_id=current_user.id,
                conversation_id=conversation_id,
                query=req.query,
                knowledge_bool=req.knowledge_bool,
            ):
                # 检查客户端是否断开连接
                if await request.is_disconnected():
                    logger.info(f"[CLIENT DISCONNECTED] 用户 {current_user.username} 断开连接，取消正在运行的任务")
                    await svc.cancel_runs(thread_id=current_user.id, conversation_id=conversation_id)
                    break
                
                # v2 格式：{"type": "...", "ns": [...], "data": ...}。
                # metadata/partial/updates 是 SDK 事件，不是最终前端协议。
                event_type = event.get("type", "")
                event_data = event.get("data", {})
                # updates 承载 interrupt，因为节点暂停时没有普通 answer；
                # 必须单独发送 interrupt，让前端显示澄清/审阅控件。
                if event_type == "updates" and isinstance(event_data, dict):
                    for node_name, node_output in event_data.items():
                        if isinstance(node_output, dict):
                            sources = node_output.get("sources", [])
                            if sources:
                                final_sources = sources
                            
                            # 检查是否有 answer 内容需要发送
                            answer = node_output.get("answer", "")
                            if (
                                answer
                                and node_name in DISPLAY_NODES
                                and node_name not in streamed_display_nodes
                            ):
                                # updates 仅作为没有 token 流事件时的兜底，避免重复发送完整答案
                                yield sse_event("content", content=answer, is_display=True)
                    
                    if "__interrupt__" in event_data:
                        interrupt_data = normalize_interrupt_payload(event_data["__interrupt__"])
                        # 发送会话 ID
                        yield sse_event("conversation_id", content=conversation_id)
                        yield sse_event("interrupt", content=interrupt_data)
                        return
                
                # metadata 只负责确定当前节点和发送进度事件，不把 metadata
                # 中的内部运行信息直接暴露给前端。
                if event_type == "messages/metadata" and isinstance(event_data, dict):
                    for run_id, run_info in event_data.items():
                        if isinstance(run_info, dict):
                            metadata = run_info.get("metadata", {})
                            node_name = metadata.get("langgraph_node", "")
                            if node_name and node_name != current_node:
                                current_node = node_name
                                prev_content = ""  # 重置内容追踪
                                # 发送节点开始事件
                                if node_name in TRACK_NODES:
                                    label = NODE_LABELS.get(node_name, f"正在执行 {node_name}...")
                                    yield sse_event(
                                        "node_start",
                                        node=node_name,
                                        label=label,
                                        is_display=node_name in DISPLAY_NODES,
                                    )
                    continue
                
                # partial 才是 token/累计文本流。根据当前节点选择 article
                # （草稿）或 content（最终对话答案）协议。
                if event_type == "messages/partial":
                    # current_node 为空时，不发送 content 事件
                    if not current_node:
                        continue
                    
                    # draft 节点的内容使用 article 事件类型发送
                    if current_node == 'draft':
                        if isinstance(event_data, list):
                            for msg_chunk in event_data:
                                if isinstance(msg_chunk, dict):
                                    content = msg_chunk.get("content", "")
                                    if content and isinstance(content, str):
                                        if prev_content and content.startswith(prev_content):
                                            delta = content[len(prev_content):]
                                        else:
                                            delta = content
                                        
                                        if delta:
                                            yield sse_event("article", content=delta)
                                        prev_content = content
                    # 显示节点的内容使用 content 事件类型发送
                    elif current_node in DISPLAY_NODES:
                        if isinstance(event_data, list):
                            for msg_chunk in event_data:
                                if isinstance(msg_chunk, dict):
                                    content = msg_chunk.get("content", "")
                                    if content and isinstance(content, str):
                                        if prev_content and content.startswith(prev_content):
                                            delta = content[len(prev_content):]
                                        else:
                                            delta = content
                                        
                                        if delta:
                                            yield sse_event("content", content=delta, is_display=True)
                                            streamed_display_nodes.add(current_node)
                                        prev_content = content

            # 发送会话 ID
            yield sse_event("conversation_id", content=conversation_id)

            # 发送最终 sources（如果有）
            if final_sources:
                formatted_sources = normalize_sources(final_sources)
                yield sse_event("sources", content=formatted_sources or final_sources)

            # 发送完成信号
            yield sse_event("complete", finished=True)

        except asyncio.CancelledError:
            logger.info(f"[STREAM CANCELLED] 用户 {current_user.username} 的流式请求被取消，取消 LangGraph 任务")
            try:
                await asyncio.shield(svc.cancel_runs(thread_id=current_user.id, conversation_id=conversation_id))
            except asyncio.CancelledError:
                logger.info(f"[SHIELD CANCELLED] shield 被取消，但 cancel_runs 应该继续执行")
        except Exception as e:
            logger.error(f"[STREAM ERROR] {str(e)}")
            error_data = {
                "type": "error",
                "content": f"流式传输错误: {str(e)}",
                "finished": True,
            }
            yield sse_event("error", content=error_data["content"], finished=True)

    return StreamingResponse(
        content=event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@router.get("/history")
async def get_user_history(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    svc: LangGraphService = Depends(get_langgraph_service),
) -> List[ChatMessage]:
    """
    获取当前用户指定会话的历史消息。

    这里读取的是 LangGraph thread state 里的 messages，
    也就是"短期记忆"部分，而不是你额外存到 PostgreSQL 长期记忆表里的内容。
    """
    # history 读取的是 checkpoint 中的短期消息；长期记忆不是聊天记录，不在
    # 这里拼接，避免把内部上下文重复展示给用户。
    try:
        chat_history = await svc.get_history(
            thread_id=current_user.id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

    result: list[ChatMessage] = []
    for msg in chat_history:
        role = msg.get("role") or msg.get("type") or "ai"
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        # 从消息元数据中提取sources，实现消息级别的sources关联
        sources = msg.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        result.append(ChatMessage(role=str(role), content=str(content), sources=sources))
    return result


@router.post("/clear", response_model=CommonResponse)
async def chat_clear(
    req: ClearRequest,
    current_user: User = Depends(get_current_active_user),
    svc: LangGraphService = Depends(get_langgraph_service),
):
    """
    清空指定会话的 LangGraph 短期记忆。

    注意：
    - 当前只清理 LangGraph thread 状态
    - 不删除 PostgreSQL 长期记忆表中的历史沉淀
    """
    try:
        await svc.clear_history(
            thread_id=current_user.id,
            conversation_id=req.conversation_id,
        )
        return CommonResponse(status="success", message="会话已清空")
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=CommonResponse)
async def chat_clear(
    _: ClearRequest,
    current_user: User = Depends(get_current_active_user),
    svc: LangGraphService = Depends(get_langgraph_service),
):
    """
    清空当前用户的 LangGraph 短期记忆。

    注意：
    - 当前只清理 LangGraph thread 状态
    - 不删除 PostgreSQL 长期记忆表中的历史沉淀
    """
    try:
        await svc.clear_history(thread_id=current_user.id)
        return CommonResponse(status="success", message="会话已清空")
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/resume/stream")
async def chat_resume_stream(
    req: ResumeRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    svc: LangGraphService = Depends(get_langgraph_service),
):
    """
    流式恢复被 interrupt 暂停的写作流程。
    """
    async def event_generator():
        # resume 与普通 stream 使用同一套事件翻译，但输入由前端动作转换成
        # Command(resume=...)；恢复后仍可能再次遇到 interrupt，因此循环中
        # 继续监听并转发 interrupt。
        try:
            yield sse_event(
                "status",
                content="正在继续处理...",
                is_display=False,
            )
            final_sources = []
            prev_content = ""
            current_node = ""
            # 只有这些节点的内容会显示在对话中

            async for event in svc.resume_stream(
                thread_id=current_user.id,
                conversation_id=req.conversation_id,
                action=req.action,
                content=req.content,
                answer=req.answer,
            ):
                # 检查客户端是否断开连接
                if await request.is_disconnected():
                    logger.info(f"[CLIENT DISCONNECTED] 用户 {current_user.username} 断开连接，取消正在运行的任务")
                    await svc.cancel_runs(thread_id=current_user.id, conversation_id=req.conversation_id)
                    break
                
                event_type = event.get("type", "")
                event_data = event.get("data", {})
                
                # 检测 interrupt 事件（resume 后可能再次遇到人机交互）
                if event_type == "updates" and isinstance(event_data, dict):
                    for node_name, node_output in event_data.items():
                        if isinstance(node_output, dict):
                            sources = node_output.get("sources", [])
                            if sources:
                                final_sources = sources

                    if "__interrupt__" in event_data:
                        interrupt_data = normalize_interrupt_payload(event_data["__interrupt__"])
                        yield sse_event("interrupt", content=interrupt_data)
                        return

                # 通过 messages/metadata 事件判断进入哪个节点
                if event_type == "messages/metadata" and isinstance(event_data, dict):
                    for run_id, run_info in event_data.items():
                        if isinstance(run_info, dict):
                            metadata = run_info.get("metadata", {})
                            node_name = metadata.get("langgraph_node", "")
                            if node_name and node_name != current_node:
                                current_node = node_name
                                prev_content = ""
                                if node_name in TRACK_NODES:
                                    label = NODE_LABELS.get(node_name, f"正在执行 {node_name}...")
                                    yield sse_event(
                                        "node_start",
                                        node=node_name,
                                        label=label,
                                        is_display=node_name in DISPLAY_NODES,
                                    )
                    continue

                # 处理 messages/partial 事件
                if event_type == "messages/partial":
                    # current_node 为空时，不发送 content 事件
                    if not current_node:
                        continue
                    
                    # draft 节点的内容使用 article 事件类型发送
                    if current_node == 'draft':
                        if isinstance(event_data, list):
                            for msg_chunk in event_data:
                                if isinstance(msg_chunk, dict):
                                    content = msg_chunk.get("content", "")
                                    if content and isinstance(content, str):
                                        if prev_content and content.startswith(prev_content):
                                            delta = content[len(prev_content):]
                                        else:
                                            delta = content

                                        if delta:
                                            yield sse_event("article", content=delta)
                                        prev_content = content
                    # 显示节点的内容使用 content 事件类型发送
                    elif current_node in DISPLAY_NODES:
                        if isinstance(event_data, list):
                            for msg_chunk in event_data:
                                if isinstance(msg_chunk, dict):
                                    content = msg_chunk.get("content", "")
                                    if content and isinstance(content, str):
                                        if prev_content and content.startswith(prev_content):
                                            delta = content[len(prev_content):]
                                        else:
                                            delta = content

                                        if delta:
                                            yield sse_event("content", content=delta, is_display=True)
                                        prev_content = content

            if final_sources:
                formatted_sources = normalize_sources(final_sources)
                yield sse_event("sources", content=formatted_sources or final_sources)

            yield sse_event("complete", finished=True)

        except asyncio.CancelledError:
            logger.info(f"[STREAM CANCELLED] 用户 {current_user.username} 的流式请求被取消，取消 LangGraph 任务")
            try:
                await asyncio.shield(svc.cancel_runs(thread_id=current_user.id, conversation_id=req.conversation_id))
            except asyncio.CancelledError:
                logger.info(f"[SHIELD CANCELLED] shield 被取消，但 cancel_runs 应该继续执行")
        except Exception as e:
            logger.error(f"[STREAM ERROR] {str(e)}")
            error_data = {
                "type": "error",
                "content": f"流式传输错误: {str(e)}",
                "finished": True,
            }
            yield sse_event("error", content=error_data["content"], finished=True)

    return StreamingResponse(
        content=event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
