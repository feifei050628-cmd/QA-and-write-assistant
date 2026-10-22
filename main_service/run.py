from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# 根目录 .env 里的 MAIN_SERVICE_PORT 要在 uvicorn.run 之前生效，因此这里先加载。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# reload=True 会派生子进程，这里设置 PYTHONUTF8 让子进程同样使用 UTF-8，
# 避免含 emoji 的模型输出在 GBK 控制台上抛 UnicodeEncodeError。
os.environ.setdefault("PYTHONUTF8", "1")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


if __name__ == "__main__":
    # 开发启动入口。reload=True 适合本地调试，生产环境应由进程管理器或
    # 单独的 Uvicorn 命令控制 worker 数量和重启策略。
    # 端口可用环境变量 MAIN_SERVICE_PORT 覆盖：本机 8000 被其他项目占用时，
    # 在根目录 .env 中设置 MAIN_SERVICE_PORT=8002 即可，无需修改代码。
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=int(os.getenv("MAIN_SERVICE_PORT", "8000")),
        reload=True,
        log_level="info",
    )
