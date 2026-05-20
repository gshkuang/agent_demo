# agent_demo - 生产级 AI Agent 实验项目

基于 LangGraph/LangChain 的 Agent 实验集合，覆盖从原型到生产的完整链路。

## 项目结构

```
agent_demo/
├── api/                        # P0: FastAPI 生产服务
│   ├── main.py                 # 主服务入口
│   ├── Dockerfile              # 容器化
│   ├── docker-compose.yml      # 完整编排 (服务 + Redis + Prometheus + Grafana + Jaeger)
│   ├── requirements-api.txt    # API 依赖
│   └── .dockerignore
├── observability/              # P0: 可观测性
│   ├── tracing.py              # OpenTelemetry 分布式追踪
│   └── metrics.py              # Prometheus 指标收集
├── planning/                   # ReAct / Reflexion Agent (LangGraph)
├── memory/                     # 记忆系统 (LangChain + Chroma + SQLite)
├── tool_safety/               # 工具安全 (BaseTool + Pydantic + 审计)
├── multi_agent/               # Manager-Worker (LangGraph Send)
├── evaluation/                # 三维评估 (LangSmith + CriteriaEvalChain)
├── rag-experiments/           # RAG实验 (保持独立)
├── mcp_server/                # P1: MCP Server (FastMCP)
│   └── stock_mcp_server.py    # 股票分析 MCP 服务
├── frameworks/                # P2: 多框架对比
│   ├── autogen_demo.py        # Microsoft AutoGen
│   ├── crewai_demo.py         # CrewAI 角色驱动
│   └── adk_demo.py            # Google ADK
├── distributed/               # P3: 分布式架构
│   ├── celery_worker.py       # Celery + Redis 任务队列
│   └── redis_state.py         # Redis 状态共享 + 分布式锁
├── computer_use/              # P3: Computer Use Agent
│   ├── browser_agent.py       # Playwright 浏览器自动化
│   └── code_agent.py          # 沙箱代码执行
└── data/                      # 数据目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 添加 API Key
```

### 3. 启动服务

**本地开发:**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**Docker 部署:**
```bash
cd api
docker-compose up -d
```

**访问:**
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Jaeger: http://localhost:16686

### 4. 启动 Celery Worker

```bash
celery -A distributed.celery_worker worker --loglevel=info --concurrency=2
```

### 5. 启动 MCP Server

```bash
cd mcp_server
python stock_mcp_server.py --transport stdio
```

## 面试考点覆盖

| 面试题 | 模块 | 文件 | 优先级 |
|--------|------|------|--------|
| ReAct vs Plan-and-Execute | planning | react_agent.py | ✅ |
| 工具安全三层防护 | tool_safety | safe_tool_executor.py | ✅ |
| 多Agent协作 | multi_agent | manager_worker.py | ✅ |
| 记忆系统设计 | memory | memory_system.py | ✅ |
| Agent三维评估 | evaluation | agent_evaluator.py | ✅ |
| RAG系统设计 | rag-experiments | 独立实验 | ✅ |
| **生产级部署** | **api** | **main.py + Dockerfile** | **P0** |
| **可观测性** | **observability** | **tracing.py + metrics.py** | **P0** |
| **MCP Server** | **mcp_server** | **stock_mcp_server.py** | **P1** |
| **AutoGen** | **frameworks** | **autogen_demo.py** | **P2** |
| **CrewAI** | **frameworks** | **crewai_demo.py** | **P2** |
| **Google ADK** | **frameworks** | **adk_demo.py** | **P2** |
| **分布式任务** | **distributed** | **celery_worker.py** | **P3** |
| **状态共享** | **distributed** | **redis_state.py** | **P3** |
| **浏览器自动化** | **computer_use** | **browser_agent.py** | **P3** |
| **代码执行** | **computer_use** | **code_agent.py** | **P3** |

## 技术栈

- **LangGraph**: Agent 工作流编排
- **LangChain**: 工具包装、Memory、评估
- **FastAPI**: REST API 服务
- **Docker**: 容器化部署
- **Prometheus + Grafana**: 监控
- **OpenTelemetry + Jaeger**: 分布式追踪
- **Celery + Redis**: 异步任务队列
- **MCP**: Model Context Protocol
- **Playwright**: 浏览器自动化

## RAG实验

详见 [rag-experiments/README.md](./rag-experiments/README.md)

| 模块 | 最佳方案 |
|------|---------|
| 文本分块 | 递归分块 / 结构化感知 |
| 向量嵌入 | bge-large-zh-v1.5 |
| 检索策略 | 混合RRF |
| 端到端 | recursive + bge-large-zh + hybrid_rrf |

---
*维护者: Ethon*
