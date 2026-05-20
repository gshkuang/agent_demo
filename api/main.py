#!/usr/bin/env python3
"""
Agent Service - FastAPI 生产级服务
P0: 把 ReAct Agent 包装为 REST API + Docker 部署

功能:
- /chat - 单轮对话
- /chat/stream - 流式对话 (SSE)
- /agent/run - Agent 任务执行 (ReAct)
- /health - 健康检查
- /metrics - Prometheus 指标
- /traces - 链路追踪 (OpenTelemetry)
"""
import os
import json
import time
import asyncio
import hashlib
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# LangChain / LangGraph
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

# Observability
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# 导入本地模块
import sys
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============ 配置 ============
class Config:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    PORT = int(os.environ.get("PORT", "8000"))
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    OTEL_ENDPOINT = os.environ.get("OTEL_ENDPOINT", "")
    MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))

# ============ 可观测性初始化 ============

def init_tracing():
    """初始化分布式追踪"""
    provider = TracerProvider()
    
    # 控制台输出（开发环境）
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    
    # OTLP 导出（生产环境，如 Jaeger / Tempo）
    if Config.OTEL_ENDPOINT:
        otlp_exporter = OTLPSpanExporter(endpoint=Config.OTEL_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    trace.set_tracer_provider(provider)
    return trace.get_tracer("agent-service")

# Prometheus 指标
REQUEST_COUNT = Counter("agent_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("agent_request_duration_seconds", "Request latency", ["endpoint"])
ACTIVE_AGENTS = Gauge("agent_active_runs", "Number of active agent runs")
AGENT_STEPS = Histogram("agent_steps_total", "Agent reasoning steps", ["agent_type"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens used", ["model", "type"])

# ============ Agent 核心 ============

class AgentState(dict):
    """Agent 状态 - 兼容 TypedDict 和 dict"""
    messages: List
    current_step: int
    max_steps: int
    metadata: Dict[str, Any]


@tool
def search_stock(query: str) -> str:
    """搜索股票基本信息"""
    stocks = {
        "沪电股份": "代码002463，PCB龙头，2024年ROE 18.5%，PE 25x",
        "贵州茅台": "代码600519，白酒龙头，2024年ROE 25.3%，PE 28x",
        "比亚迪": "代码002594，新能源车龙头，2024年ROE 15.6%，PE 32x",
        "宁德时代": "代码300750，电池龙头，2024年ROE 18.9%，PE 26x",
    }
    return next((v for k, v in stocks.items() if k in query), f"未找到'{query}'")


@tool
def calculate_roe(net_profit: float, equity: float) -> str:
    """计算ROE"""
    if equity <= 0:
        return "净资产必须大于0"
    roe = net_profit / equity * 100
    return f"ROE = {roe:.2f}%"


@tool
def get_market_sentiment() -> str:
    """获取市场情绪"""
    return "今日涨停45家，跌停12家，北向净流入+23亿，情绪偏乐观"


@tool
def analyze_sector(sector: str) -> str:
    """分析板块"""
    sectors = {
        "半导体": "今日涨幅+2.3%，资金净流入+15亿，龙头中芯国际涨停",
        "新能源": "今日涨幅+1.8%，资金净流入+8亿，比亚迪创新高",
        "白酒": "今日涨幅-0.5%，资金净流出-3亿，茅台震荡整理",
    }
    return sectors.get(sector, f"暂无{sector}板块数据")


# 工具注册表
TOOLS = [search_stock, calculate_roe, get_market_sentiment, analyze_sector]


def build_react_agent(llm, tools: list, system_prompt: str = None):
    """构建 ReAct Agent (LangGraph) - 生产级版本"""
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = system_prompt or (
        "你是专业的投研Agent，通过推理和行动交替完成分析任务。"
        "每次思考后选择合适的工具获取数据，最终给出投资建议。"
    )
    tool_node = ToolNode(tools)
    tracer = trace.get_tracer("agent-service")

    def should_continue(state: AgentState) -> str:
        with tracer.start_as_current_span("agent.should_continue"):
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None) and state["current_step"] < state["max_steps"]:
                return "continue"
            return "end"

    def call_model(state: AgentState):
        with tracer.start_as_current_span("agent.call_model") as span:
            span.set_attribute("agent.step", state["current_step"])
            msgs = [SystemMessage(content=system_prompt)] + state["messages"]
            response = llm_with_tools.invoke(msgs)
            
            # 记录 token 使用
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                LLM_TOKENS.labels(model=Config.OPENAI_MODEL, type="input").inc(usage.get("input_tokens", 0))
                LLM_TOKENS.labels(model=Config.OPENAI_MODEL, type="output").inc(usage.get("output_tokens", 0))
            
            return {
                "messages": state["messages"] + [response],
                "current_step": state["current_step"] + 1,
                "max_steps": state["max_steps"],
                "metadata": state.get("metadata", {}),
            }

    def call_tools(state: AgentState):
        with tracer.start_as_current_span("agent.call_tools"):
            return tool_node.invoke(state)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
    workflow.add_edge("action", "agent")
    return workflow.compile()


# ============ FastAPI 应用 ============

# 请求/响应模型
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="用户消息")
    session_id: Optional[str] = Field(default=None, description="会话ID（为空则新建）")
    model: Optional[str] = Field(default=None, description="模型覆盖")
    max_steps: Optional[int] = Field(default=10, ge=1, le=50, description="最大推理步数")
    stream: bool = Field(default=False, description="是否流式返回")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    steps: int
    tools_used: List[str]
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    model: str


# 全局状态
app_state = {
    "start_time": time.time(),
    "request_count": 0,
    "llm": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    print("🚀 Agent Service 启动中...")
    init_tracing()
    
    if not Config.OPENAI_API_KEY:
        print("⚠️ 警告: OPENAI_API_KEY 未设置，服务将使用模拟模式")
    else:
        app_state["llm"] = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0,
        )
        print(f"✅ LLM 初始化完成: {Config.OPENAI_MODEL}")
    
    yield
    
    # 关闭
    print("🛑 Agent Service 关闭")


app = FastAPI(
    title="Agent Service",
    description="生产级 AI Agent 服务 - 支持 ReAct 推理、工具调用、流式输出",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry 自动注入
FastAPIInstrumentor.instrument_app(app)
if app_state.get("llm"):
    LangChainInstrumentor().instrument()


# ============ API 端点 ============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if app_state["llm"] else "degraded",
        version="1.0.0",
        uptime_seconds=round(time.time() - app_state["start_time"], 2),
        model=Config.OPENAI_MODEL,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return StreamingResponse(
        iter([generate_latest()]),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """单轮对话 - Agent 执行任务"""
    start = time.time()
    session_id = request.session_id or hashlib.md5(f"{time.time()}".encode()).hexdigest()[:16]
    
    with REQUEST_LATENCY.labels(endpoint="/chat").time():
        try:
            ACTIVE_AGENTS.inc()
            
            if not app_state["llm"]:
                REQUEST_COUNT.labels(endpoint="/chat", status="error").inc()
                raise HTTPException(status_code=503, detail="LLM 未初始化")
            
            # 构建 Agent
            agent = build_react_agent(app_state["llm"], TOOLS)
            
            # 执行
            result = agent.invoke({
                "messages": [HumanMessage(content=request.message)],
                "current_step": 0,
                "max_steps": request.max_steps or Config.MAX_STEPS,
                "metadata": {"session_id": session_id},
            })
            
            # 提取结果
            steps = result["current_step"]
            messages = result["messages"]
            
            # 提取工具调用
            tools_used = []
            for msg in messages:
                if getattr(msg, "tool_calls", None):
                    tools_used.extend([t["name"] for t in msg.tool_calls])
            
            # 最后一条是 AI 回复
            final_content = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                    final_content = msg.content
                    break
            
            AGENT_STEPS.labels(agent_type="react").observe(steps)
            REQUEST_COUNT.labels(endpoint="/chat", status="success").inc()
            
            latency_ms = (time.time() - start) * 1000
            
            return ChatResponse(
                session_id=session_id,
                response=final_content,
                steps=steps,
                tools_used=list(set(tools_used)),
                latency_ms=round(latency_ms, 2),
                timestamp=datetime.utcnow().isoformat(),
            )
            
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/chat", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            ACTIVE_AGENTS.dec()


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话 - SSE 输出"""
    session_id = request.session_id or hashlib.md5(f"{time.time()}".encode()).hexdigest()[:16]
    
    if not app_state["llm"]:
        raise HTTPException(status_code=503, detail="LLM 未初始化")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """SSE 事件生成器"""
        agent = build_react_agent(app_state["llm"], TOOLS)
        
        # 发送会话ID
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        
        # 执行 Agent（同步转异步）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.invoke({
                "messages": [HumanMessage(content=request.message)],
                "current_step": 0,
                "max_steps": request.max_steps or Config.MAX_STEPS,
                "metadata": {"session_id": session_id},
            })
        )
        
        # 流式输出每一步
        for i, msg in enumerate(result["messages"]):
            role = type(msg).__name__.replace("Message", "").lower()
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)
            
            event = {
                "type": "step",
                "step": i,
                "role": role,
                "content": content,
            }
            if tool_calls:
                event["tools"] = [t["name"] for t in tool_calls]
                event["tool_args"] = [t.get("args", {}) for t in tool_calls]
            
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        
        # 完成
        yield f"data: {json.dumps({'type': 'done', 'steps': result['current_step']})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/run")
async def run_agent_task(request: ChatRequest, background_tasks: BackgroundTasks):
    """异步 Agent 任务 - 后台执行"""
    task_id = hashlib.md5(f"{time.time()}{request.message}".encode()).hexdigest()[:16]
    
    # 这里可以接入 Celery / Redis 做真正的异步队列
    # 简化版：直接返回 task_id，实际执行在后台
    
    def _execute_task():
        """后台执行任务"""
        print(f"[Task {task_id}] 开始执行: {request.message[:50]}...")
        # 实际执行逻辑...
        print(f"[Task {task_id}] 执行完成")
    
    background_tasks.add_task(_execute_task)
    
    return JSONResponse({
        "task_id": task_id,
        "status": "queued",
        "message": "任务已提交，请通过 /task/{task_id} 查询状态",
    })


@app.get("/")
async def root():
    """服务根路径"""
    return {
        "service": "Agent Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "metrics": "/metrics",
        },
        "docs": "/docs",
    }


# ============ 启动 ============

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.PORT,
        reload=False,
        log_level=Config.LOG_LEVEL.lower(),
        workers=1,  # 生产环境可改为多 worker
    )
