"""FastAPI 应用入口。

lifespan 负责一次性的数据库初始化和下游 HTTP 客户端生命周期；路由注册
则把用户、文档、旧 LangGraph 代理和自定义 LangGraph 代理暴露给前端。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.routers import chat, documents, users, custom_chat
from db import Base, engine, SessionLocal
from models import UserModel
from app.services.auth_service import get_password_hash
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _ensure_role_column():
    """兼容旧数据库：如果 users 表缺少 role 列则自动添加。"""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='users' AND column_name='role'"
        ))
        if result.fetchone() is None:
            logger.info("users 表缺少 role 列，正在添加...")
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN role VARCHAR(16) NOT NULL DEFAULT 'user'"
            ))
            conn.commit()
            logger.info("role 列添加成功")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 新库先按 ORM 模型建表；旧库再补充可能缺失的 role 列。
    # 这里没有使用迁移框架，因此只做当前项目启动所需的最小兼容初始化。
    Base.metadata.create_all(bind=engine)
    _ensure_role_column()
    db = SessionLocal()
    try:
        root_user = db.query(UserModel).filter(UserModel.username == "root").first()
        if root_user is None:
            db.add(
                UserModel(
                    username="root",
                    full_name="Administrator",
                    email="admin@example.com",
                    hashed_password=get_password_hash("admin123"),
                    disabled=False,
                    role="admin",
                )
            )
            db.commit()
        else:
            # 确保 root 用户是 admin 角色
            if root_user.role != "admin":
                root_user.role = "admin"
                db.commit()
    finally:
        db.close()
    # 自定义聊天路由复用一个进程级 httpx 客户端，先在应用启动时建立，
    # 避免第一条聊天请求承担连接初始化开销。
    await custom_chat.startup_custom_chat()
    try:
        yield
    finally:
        # 先停止下游客户端，再结束 FastAPI lifespan，确保流式请求使用的
        # 连接池能够正常关闭。
        await custom_chat.shutdown_custom_chat()


app = FastAPI(title="Agent API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/docs", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(custom_chat.router, prefix="/api/custom-chat", tags=["custom-chat"])

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "healthy"}
