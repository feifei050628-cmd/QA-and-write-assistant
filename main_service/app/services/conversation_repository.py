"""业务会话表的 CRUD 仓储。

会话 ID 是 main_service 对外使用的稳定标识。真正传给 LangGraph 的 thread
ID由下游适配层根据用户 ID和会话 ID重新映射，因此这里不直接保存 checkpoint。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from models import ConversationModel


class ConversationRepository:
    """
    会话数据访问层。

    管理用户会话的 CRUD 操作。
    """

    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: str, title: str = "新对话") -> ConversationModel:
        """创建新会话。"""
        conversation = ConversationModel(
            user_id=user_id,
            title=title,
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get(self, conversation_id: str, user_id: str) -> ConversationModel | None:
        """获取指定会话，并把 user_id 作为查询条件完成归属校验。"""
        return self.db.query(ConversationModel).filter(
            ConversationModel.id == conversation_id,
            ConversationModel.user_id == user_id,
        ).first()

    def list_by_user(self, user_id: str) -> list[ConversationModel]:
        """获取用户的所有会话，按更新时间倒序。"""
        return list(
            self.db.query(ConversationModel)
            .filter(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.updated_at.desc())
            .all()
        )

    def update_title(self, conversation_id: str, user_id: str, title: str) -> ConversationModel | None:
        """更新会话标题。"""
        conversation = self.get(conversation_id, user_id)
        if conversation is None:
            return None
        conversation.title = title
        conversation.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def delete(self, conversation_id: str, user_id: str) -> bool:
        """删除会话。"""
        conversation = self.get(conversation_id, user_id)
        if conversation is None:
            return False
        self.db.delete(conversation)
        self.db.commit()
        return True

    def touch(self, conversation_id: str, user_id: str) -> None:
        """更新会话的最后活跃时间。

        路由在复用已有会话发起聊天前调用此方法，使会话列表可以按最近活动
        排序；不存在或不属于当前用户的会话不会被修改。
        """
        conversation = self.get(conversation_id, user_id)
        if conversation is not None:
            conversation.updated_at = datetime.utcnow()
            self.db.commit()
