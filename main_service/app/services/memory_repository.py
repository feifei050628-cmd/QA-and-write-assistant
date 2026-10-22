"""main_service 自己的简单长期记忆仓储。

它是独立的数据库模型，当前聊天主流程主要使用 langgraph_service 的
AsyncPostgresStore；保留本仓储是为了管理 main_service 侧的显式记忆条目，
两者不要混淆成同一种存储。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from models import LongTermMemoryModel


class MemoryRepository:
    """
    长期记忆数据访问层。

    当前设计很简单：
    - add_memory: 写入一条长期记忆
    - list_memories: 查询某个用户最近的若干条长期记忆

    这部分与 LangGraph checkpoint 的短期记忆不同，
    它更适合做长期沉淀、偏好记录、用户背景记录等。
    """

    def __init__(self, db: Session):
        self.db = db

    def add_memory(self, *, user_id: str, content: str) -> LongTermMemoryModel:
        """为指定用户写入一条长期记忆。"""
        memory = LongTermMemoryModel(user_id=user_id, content=content)
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        return memory

    def list_memories(self, *, user_id: str, limit: int = 10) -> list[LongTermMemoryModel]:
        """获取某个用户最近的长期记忆。"""
        return list(
            self.db.query(LongTermMemoryModel)
            .filter(LongTermMemoryModel.user_id == user_id)
            .order_by(LongTermMemoryModel.created_at.desc())
            .limit(limit)
            .all()
        )
