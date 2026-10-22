"""自定义聊天 API 的请求模型。

把模型单独放在 ``schemas`` 包中，可以让路由文件只关注 HTTP 流程，
也便于后续为不同接口增加响应模型，而不把数据结构和业务代码混在一起。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """一次业务聊天请求。"""

    # user_id 参与长期记忆 namespace；同一个用户可以有多个会话，但不会
    # 读取其他用户的长期记忆。
    user_id: str = Field(min_length=1, description="业务用户 ID，用于隔离长期记忆")
    # conversation_id 映射为 checkpointer 的 thread_id。为空时由路由创建新会话，
    # 后续必须把返回的会话 ID带回，才能继续同一段短期上下文。
    conversation_id: str | None = Field(default=None, description="业务会话 ID")
    # message 是当前轮新增的人类消息，不要传整个历史；历史由 PostgreSQL
    # checkpointer 根据 thread_id 自动恢复。
    message: str = Field(min_length=1, description="本轮用户消息")
    # None 表示遵循图内部默认策略，True/False 才是显式覆盖知识库路由。
    knowledge_bool: bool | None = Field(default=None, description="是否启用知识库检索")


class ResumeRequest(BaseModel):
    """恢复 LangGraph interrupt 时提交的业务动作。"""

    # resume 必须使用原 conversation_id，LangGraph 才能找到中断 checkpoint。
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    # action 会被转换成 Command(resume=...) 传给 interrupt 所在节点。
    action: str = Field(min_length=1, description="approve、edit、rewrite 或 clarify")
    content: str | None = None
    answer: str | None = None


class CancelRequest(BaseModel):
    """取消请求；run_id 为空时取消该会话当前全部任务。

    取消只终止运行中的 asyncio Task，不会删除 checkpoint，也不会删除用户的
    长期记忆；因此用户仍可以读取历史，或在适当位置继续恢复中断流程。
    """

    # 取消只作用于当前 FastAPI worker 内登记的 asyncio Task，不会删除历史。
    user_id: str = Field(min_length=1)
    conversation_id: str | None = None
    run_id: str | None = None
