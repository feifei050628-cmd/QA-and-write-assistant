from __future__ import annotations

import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import Settings
from rest_api import router as rest_router

# Windows 控制台默认 GBK；RAG 流程会 print 模型回答，遇到 emoji 会抛
# UnicodeEncodeError。统一改成 UTF-8，避免调试输出打断业务请求。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

app = FastAPI(title="RAG API Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest_router, prefix="/api/docs")

if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host=Settings.HOST,
        port=Settings.PORT,
        reload=False,
    )
