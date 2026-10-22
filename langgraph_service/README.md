# langgraph_service 说明文档

## 1. 服务定位

`langgraph_service` 是系统中的多智能体编排服务，负责：

- 维护 LangGraph 对话图
- 管理短期记忆与状态恢复
- 进行意图识别与多节点路由
- 通过 MCP over HTTP 调用 `mcp_service.query_rag` 获取知识库能力

方案二的自定义 FastAPI 服务直接提供业务 API，供 `main_service` 调用；原有
LangGraph CLI 图仍然作为备用入口保留。

---

## 2. 目录结构

```text
langgraph_service/
├── my_agent/
│   ├── agent.py
│   ├── utils/
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── tools.py
│   └── writing_subgraph/
│       ├── graph.py
│       ├── nodes.py
│       └── state.py
├── config/
│   └── settings.py
├── app/
│   ├── main.py             # FastAPI 应用对象和 lifespan
│   ├── api/
│   │   └── chat.py         # 聊天、流式、恢复、历史和取消接口
│   ├── schemas/
│   │   └── chat.py         # 聊天接口的 Pydantic 请求模型
│   └── services/
│       └── graph_runtime.py # PostgreSQL + LangGraph 图运行时
├── run.py                  # Uvicorn 启动入口
├── langgraph_app.py        # 旧图导出包装
├── main.py                 # 项目模板入口，当前部署不使用
├── langgraph.json
├── pyproject.toml        # 依赖清单（uv 管理）
└── uv.lock               # 依赖锁定文件
```

说明：
- `my_agent/agent.py`：图构建入口
- `my_agent/utils/nodes.py`：各个节点逻辑
- `my_agent/utils/tools.py`：MCP 工具调用封装
- `my_agent/writing_subgraph/`：写作子图（包含 `writing_agent` 和 `review_agent`）
- `langgraph.json`：LangGraph CLI 配置文件
- `app/api/chat.py`：方案二的 `/api/*` 业务接口，不实现 `/threads`、`/runs` 等 SDK 协议
- `app/schemas/chat.py`：请求体模型，负责校验 `user_id`、`message` 等字段
- `app/services/graph_runtime.py`：构建带 PostgreSQL 短期 checkpoint 和长期 store 的图
- `app/main.py`：创建 FastAPI 应用并管理运行时生命周期
- `run.py`：Uvicorn 进程启动文件
- `langgraph_app.py`：仅作为旧代码导出 `my_agent.agent.graph`，不是方案二入口

---

## 3. 图结构说明

当前图的大致流程为：

```text
START
  -> summarize
  -> intent_router
      -> knowledge_agent -> knowledge_guard -> fallback_search / finalize
      -> writing_workflow -> finalize
      -> chat_agent -> finalize
  -> finalize
  -> END
```

主要节点职责：

- `summarize`：短期记忆摘要（langmem SummarizationNode）
- `intent_router`：识别是知识问答、写作还是普通聊天
- `knowledge_agent`：通过 MCP 调知识库
- `knowledge_guard`：守卫节点，检测回答质量，决定是否降级
- `fallback_search`：知识库答不好时兜底搜索（Tavily）
- `writing_workflow`：写作子图，内部包含 `writing_agent` 和 `review_agent`
- `chat_agent`：普通聊天
- `finalize`：收尾，提取 answer/sources/intent 返回给调用方

---

## 4. 对外使用方式

当前方案二使用普通 HTTP 业务 API，不依赖 `langgraph-sdk`，也不实现
LangGraph Server 的 `/threads`、`/runs`、`/assistants` 协议。

`main_service` 通过 `LANGGRAPH_API_URL` 调用以下接口：

- `POST /api/chat`：非流式聊天
- `POST /api/chat/stream`：自定义 SSE 流式聊天
- `POST /api/chat/resume`：恢复 interrupt
- `GET /api/conversations/{conversation_id}/history`：读取短期历史
- `DELETE /api/conversations/{conversation_id}`：清理短期 checkpoint
- `POST /api/chat/cancel`：取消当前 worker 的任务

原有 LangGraph CLI 运行方式仍然保留，但只用于兼容或调试，不是生产部署入口。

---

## 5. 与 mcp_service 的关系

本服务中的 `knowledge_agent` 不直接访问向量库，而是通过 MCP streamable-http 调用 `mcp_service.query_rag`。

MCP 调用链路：

```text
knowledge_agent → MCP over HTTP → mcp_service:query_rag → HTTP REST → rag_api_service:query
```

调用入口在 `my_agent/utils/tools.py`。

---

## 6. 启动方式

### 6.1 原有 LangGraph CLI 方式

原有启动方式保持不变：

```bash
uv run --project langgraph_service langgraph dev --config langgraph_service/langgraph.json
```

配置文件中的图入口为：

```text
./my_agent/agent.py:graph
```

### 6.2 方案二：完全自定义业务 API

如果不需要适配 `langgraph-sdk`，可以直接使用新增的纯业务 API。它不模拟
LangGraph Server 的 `threads/runs` 协议，只暴露本项目需要的聊天、历史、恢复
和取消接口；图仍然使用 `builder.compile(checkpointer=..., store=...)` 注入
PostgreSQL 短期 checkpoint 与长期记忆。

在项目根目录启动纯业务 FastAPI 服务：

```powershell
uv run --project langgraph_service python langgraph_service/run.py
```

请求体示例：

```json
{
  "user_id": "root",
  "conversation_id": "conversation-001",
  "message": "请总结这份文档",
  "knowledge_bool": true
}
```

`main_service` 使用原有入口启动，`app.main` 中已同时注册旧聊天路由和自定义
聊天路由。自定义路由通过普通 `httpx` 请求调用上述 API：

- `POST /api/custom-chat/chat`
- `POST /api/custom-chat/chat/stream`
- `POST /api/custom-chat/chat/resume/stream`
- `GET /api/custom-chat/history?conversation_id=...`
- `POST /api/custom-chat/clear`

原来的 `main_service/run.py`、`app.main:app` 和 LangGraph CLI 入口均保留；旧的
SDK 客户端文件不参与方案二调用。取消接口使用 `asyncio.Task.cancel()`，只对当前 Uvicorn worker
有效；多 worker 部署时应把任务状态和取消信号迁移到 Redis、数据库或任务队列。
如果旧的 LangGraph CLI 服务仍占用 `2024`，可以例如使用：

```powershell
$env:LANGGRAPH_FASTAPI_PORT = "2025"
uv run --project langgraph_service python langgraph_service/run.py
```

然后将 `main_service` 的 `LANGGRAPH_API_URL` 指向对应地址，例如
`http://127.0.0.1:2025`。如果使用 FastAPI 方式替换旧 CLI 服务，则继续使用
默认的 `http://127.0.0.1:2024` 即可。

---

## 7. 关键配置

配置文件：`langgraph_service/config/settings.py`

重点环境变量：

- `POSTGRES_URL`：PostgreSQL 连接串
- `POSTGRES_URI_CUSTOM`：FastAPI 部署使用的 PostgreSQL 连接串，优先级高于
  `LANGGRAPH_POSTGRES_DSN`。当前配置示例为：
  `postgresql://postgres:123456@127.0.0.1:5432/langgraph_agents?sslmode=disable`
- `DASHSCOPE_API_KEY`：模型调用密钥
- `DASHSCOPE_BASE_URL`：模型接口地址
- `DASHSCOPE_MODEL`：默认模型
- `MODEL_TEMPERATURE`：默认温度
- `MCP_SERVICE_URL`：mcp_service MCP 地址

---

## 8. 运行依赖

1. **PostgreSQL**
   - FastAPI 启动时通过 `AsyncPostgresSaver` 保存短期记忆和线程状态
   - 通过 `AsyncPostgresStore` 注入长期记忆存储
   - 启动时会自动执行 `checkpointer.setup()` 和 `store.setup()`

2. **mcp_service**
   - 知识库能力来源（`query_rag`）
   - 需要通过 HTTP 调用 `rag_api_service`

3. **模型 API**
   - 用于对话、意图分类、写作、审校

如果 `mcp_service` 没有启动，知识问答节点会失败或退化。

### Windows 事件循环说明

Windows 默认的 `ProactorEventLoop` 与 Psycopg 异步连接不兼容。`run.py` 会
直接创建 `SelectorEventLoop` 并让 Uvicorn 在该循环中运行，不调用已经被新版
Python 标记为弃用的 `set_event_loop_policy`。因此推荐使用项目提供的 `run.py`
启动入口。

---

## 9. 调试建议

常见检查项：

1. `POSTGRES_URL` 是否可连接
2. 模型 API Key 是否配置正确
3. `mcp_service` 是否能单独启动（依赖于 `rag_api_service`）
4. `rag_api_service` 是否能单独启动
5. `langgraph.json` 中图路径是否正确（仅 CLI 部署需要）
6. `main_service` 中 `LANGGRAPH_API_URL` 是否指向本服务

补充说明：新增 FastAPI 图在每轮执行前从 `AsyncPostgresStore` 读取当前用户的
长期会话记忆，并在 `finalize` 后写入本轮问答；用户通过输入状态中的 `user_id`
隔离。原有 CLI 图和 `my_agent/utils/tools.py` 均未修改。
