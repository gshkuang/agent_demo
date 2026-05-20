#!/usr/bin/env python3
"""
Agent 可观测性 - OpenTelemetry 分布式追踪
P0: 生产级链路追踪，支持 Jaeger / Tempo / OTLP

功能:
- Agent 推理步骤追踪
- LLM 调用追踪 (token 使用、延迟)
- 工具调用追踪
- 自定义 Span 和属性
"""
import os
import time
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

# ============ 配置 ============
OTEL_ENDPOINT = os.environ.get("OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "agent-service")
OTEL_ENABLED = os.environ.get("OTEL_ENABLED", "true").lower() == "true"


class AgentTracer:
    """Agent 追踪器 - 封装 OpenTelemetry 操作"""
    
    _instance = None
    _tracer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_tracer()
        return cls._instance
    
    def _init_tracer(self):
        """初始化追踪器"""
        if not OTEL_ENABLED:
            self._tracer = None
            return
        
        resource = Resource.create({
            SERVICE_NAME: OTEL_SERVICE_NAME,
            SERVICE_VERSION: "1.0.0",
            "deployment.environment": os.environ.get("ENV", "development"),
        })
        
        provider = TracerProvider(resource=resource)
        
        # 控制台导出（开发调试）
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        
        # OTLP 导出（生产环境）
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except Exception as e:
            print(f"⚠️ OTLP 导出器初始化失败: {e}")
        
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(OTEL_SERVICE_NAME)
    
    @property
    def tracer(self):
        return self._tracer
    
    def span(self, name: str, attributes: Dict[str, Any] = None):
        """创建 Span 上下文管理器"""
        if not self._tracer:
            return _NullSpanContext()
        return self._tracer.start_as_current_span(name, attributes=attributes)
    
    def trace_agent_step(self, step_name: str, step_number: int, agent_type: str = "react"):
        """追踪 Agent 推理步骤"""
        return self.span(
            f"agent.{agent_type}.{step_name}",
            attributes={
                "agent.step_number": step_number,
                "agent.type": agent_type,
            }
        )
    
    def trace_llm_call(self, model: str, prompt_tokens: int = 0):
        """追踪 LLM 调用"""
        return self.span(
            "llm.completion",
            attributes={
                "llm.model": model,
                "llm.provider": "openai",
                "llm.prompt_tokens": prompt_tokens,
            }
        )
    
    def trace_tool_call(self, tool_name: str, tool_args: Dict[str, Any]):
        """追踪工具调用"""
        return self.span(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "tool.args": str(tool_args),
            }
        )
    
    def record_error(self, error: Exception, attributes: Dict[str, Any] = None):
        """记录错误到当前 Span"""
        if not self._tracer:
            return
        current_span = trace.get_current_span()
        current_span.set_status(Status(StatusCode.ERROR, str(error)))
        current_span.record_exception(error)
        if attributes:
            for k, v in attributes.items():
                current_span.set_attribute(k, v)


class _NullSpanContext:
    """空 Span 上下文（OTel 未启用时使用）"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_attribute(self, *args):
        pass
    def set_status(self, *args):
        pass
    def record_exception(self, *args):
        pass


def traced(name: str = None, attributes: Dict[str, Any] = None):
    """装饰器：自动追踪函数调用"""
    def decorator(func: Callable):
        span_name = name or func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = AgentTracer()
            with tracer.span(span_name, attributes=attributes or {}):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def traced_async(name: str = None, attributes: Dict[str, Any] = None):
    """装饰器：自动追踪异步函数调用"""
    def decorator(func: Callable):
        span_name = name or func.__name__
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = AgentTracer()
            with tracer.span(span_name, attributes=attributes or {}):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# ============ LangSmith 集成（可选） ============

class LangSmithTracer:
    """LangSmith 追踪集成"""
    
    def __init__(self, api_key: Optional[str] = None, project_name: str = "agent-demo"):
        self.api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
        self.project_name = project_name
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = project_name
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            print(f"✅ LangSmith 追踪已启用: {project_name}")
        else:
            print("⚠️ LangSmith API Key 未设置，追踪未启用")
    
    def trace_run(self, run_id: str, inputs: Dict, outputs: Dict, metadata: Dict = None):
        """手动记录运行（用于非 LangChain 代码）"""
        if not self.enabled:
            return
        # 实际实现会调用 LangSmith API
        pass


# ============ 演示 ============

def demo():
    """演示追踪功能"""
    print("=" * 50)
    print("Agent Tracing Demo")
    print("=" * 50)
    
    tracer = AgentTracer()
    
    # 模拟 Agent 执行流程
    with tracer.span("agent.run", attributes={"agent.type": "react", "query": "分析茅台"}):
        
        # Step 1: 理解意图
        with tracer.trace_agent_step("understand", 1):
            time.sleep(0.01)
            print("  Step 1: 理解用户意图")
        
        # Step 2: LLM 推理
        with tracer.trace_llm_call("gpt-4o", prompt_tokens=150):
            time.sleep(0.05)
            print("  Step 2: LLM 推理")
        
        # Step 3: 工具调用
        with tracer.trace_tool_call("search_stock", {"query": "贵州茅台"}):
            time.sleep(0.02)
            print("  Step 3: 调用 search_stock")
        
        # Step 4: 生成回答
        with tracer.trace_llm_call("gpt-4o", prompt_tokens=300):
            time.sleep(0.05)
            print("  Step 4: 生成最终回答")
    
    print("\n✅ 追踪数据已发送到 OTLP 端点")
    print(f"   端点: {OTEL_ENDPOINT}")
    print("   查看: http://localhost:16686 (Jaeger UI)")


if __name__ == "__main__":
    demo()
