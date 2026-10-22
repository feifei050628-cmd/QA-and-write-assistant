"""使用自定义 FastAPI LangGraph 服务的独立聊天路由。

旧路由 ``app.routers.chat`` 保持原样；本路由使用 ``/api/custom-chat`` 前缀，
便于先灰度验证，再把前端或网关切到新入口。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.routers.users import get_current_active_user
from app.schemas import ChatMessage, ChatRequest, ChatResponse, ClearRequest, ResumeRequest, User
from app.services.conversation_repository import ConversationRepository
from app.services.custom_langgraph_service import CustomLangGraphService
from db import get_db

router = APIRouter()
_service = CustomLangGraphService()

_INTERNAL_MEMORY_PREFIX = "以下是该用户的长期记忆，仅在与当前问题相关时参考："


async def startup_custom_chat() -> None:
    """预热自定义 LangGraph HTTP 连接池。"""

    await _service.startup()


async def shutdown_custom_chat() -> None:
    """释放自定义 LangGraph HTTP 连接池。"""

    await _service.shutdown()


def _conversation_id(req: ChatRequest, user: User, db: Session) -> str:
    """复用 main_service 会话表，确保新旧入口的会话列表可以共存。

    自定义入口不另建会话表；因此用户从旧入口切换到新入口时，仍可使用同一
    conversation_id，只有下游的协议适配方式发生变化。
    """

    if req.conversation_id:
        ConversationRepository(db).touch(req.conversation_id, user.username)
        return req.conversation_id
    title = req.query[:20] + "..." if len(req.query) > 20 else req.query
    return ConversationRepository(db).create(user_id=user.username, title=title).id


def _is_internal_content(content: str) -> bool:
    """过滤不应展示给用户的长期记忆上下文。"""

    return content.startswith(_INTERNAL_MEMORY_PREFIX)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)) -> ChatResponse:
    """非流式业务聊天。"""

    conversation_id = _conversation_id(req, current_user, db)
    try:
        result = await _service.chat(thread_id=current_user.id, conversation_id=conversation_id, query=req.query, knowledge_bool=req.knowledge_bool)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"自定义 LangGraph 服务调用失败: {exc}") from exc
    return ChatResponse(messages=ChatMessage(role="ai", content=str(result.get("answer", "")), sources=result.get("sources", []) or []), conversation_id=conversation_id, is_interrupt=bool(result.get("is_interrupt")))


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)) -> StreamingResponse:
    """流式业务聊天。"""

    conversation_id = _conversation_id(req, current_user, db)

    async def events():
        # LangGraph 服务已经输出统一的前端 SSE。main_service 只负责鉴权、
        # 连接管理和透传，不解析或修改下游事件。
        async for line in _service.chat_stream(thread_id=current_user.id, conversation_id=conversation_id, query=req.query, knowledge_bool=req.knowledge_bool):
            if await request.is_disconnected():
                # 连接生命周期控制，不改变下游事件内容。
                await _service.cancel_runs(thread_id=current_user.id, conversation_id=conversation_id)
                return
            yield f"{line}\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@router.get("/history")
async def history(conversation_id: str, current_user: User = Depends(get_current_active_user)) -> list[ChatMessage]:
    """读取当前用户指定会话的短期历史。"""

    messages = await _service.get_history(thread_id=current_user.id, conversation_id=conversation_id)
    result: list[ChatMessage] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        # SystemMessage 可能包含长期记忆、摘要等仅供模型使用的上下文，
        # 不能作为用户可见聊天消息返回给前端。
        # 下游返回 LangChain 的 type，旧前端模型使用 role；这里保留现有
        # 字段值，因为 ChatMessage 已经能接受 human/user/ai/assistant。
        role = str(item.get("type") or item.get("role") or "ai")
        content = str(item.get("content", ""))
        if role == "system" or _is_internal_content(content):
            continue
        additional_kwargs = item.get("additional_kwargs", {})
        sources = item.get("sources", []) or []
        if isinstance(additional_kwargs, dict):
            sources = additional_kwargs.get("sources", sources) or []
        result.append(ChatMessage(role=role, content=content, sources=sources))
    return result


@router.post("/clear")
async def clear(req: ClearRequest, current_user: User = Depends(get_current_active_user)) -> dict[str, str]:
    """清理短期 checkpoint，长期记忆不受影响。"""

    await _service.clear_history(thread_id=current_user.id, conversation_id=req.conversation_id)
    return {"status": "success", "message": "会话已清空"}


@router.post("/chat/resume/stream")
async def resume(req: ResumeRequest, current_user: User = Depends(get_current_active_user)) -> StreamingResponse:
    """恢复 interrupt 流程。"""

    async def events():
        # 恢复流同样由 LangGraph 服务生成完整 SSE，main_service 原样透传。
        async for line in _service.resume_stream(thread_id=current_user.id, conversation_id=req.conversation_id, action=req.action, content=req.content, answer=req.answer):
            yield f"{line}\n"

    return StreamingResponse(events(), media_type="text/event-stream")
