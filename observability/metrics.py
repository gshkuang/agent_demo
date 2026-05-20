#!/usr/bin/env python3
"""
Agent 指标收集 - Prometheus + 自定义指标
P0: 生产级监控指标

功能:
- 请求计数 / 延迟
- Agent 推理步数分布
- LLM Token 使用量
- 工具调用成功率
- 活跃 Agent 数量
"""
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager
from functools import wraps

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess,
)

# ============ 指标定义 ============

# 请求指标
REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "agent_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Agent 指标
ACTIVE_AGENTS = Gauge(
    "agent_active_runs",
    "Number of currently running agents"
)

AGENT_STEPS = Histogram(
    "agent_steps_total",
    "Number of reasoning steps per agent run",
    ["agent_type"],
    buckets=[1, 2, 3, 5, 8, 12, 20, 50]
)

AGENT_RUN_DURATION = Histogram(
    "agent_run_duration_seconds",
    "Agent run duration",
    ["agent_type", "status"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# LLM 指标
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "LLM token usage",
    ["model", "type"]  # type: input / output
)

LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM API latency",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "LLM API errors",
    ["model", "error_type"]
)

# 工具指标
TOOL_CALLS = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"]
)

TOOL_LATENCY = Histogram(
    "tool_call_duration_seconds",
    "Tool execution latency",
    ["tool_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# 业务指标
SESSION_COUNT = Counter(
    "agent_sessions_total",
    "Total sessions created"
)

MEMORY_OPERATIONS = Counter(
    "memory_operations_total",
    "Memory system operations",
    ["operation", "status"]
)

# 服务信息
SERVICE_INFO = Info("agent_service", "Agent service information")


class MetricsCollector:
    """指标收集器 - 封装所有指标操作"""
    
    def __init__(self):
        SERVICE_INFO.info({
            "version": "1.0.0",
            "framework": "langgraph",
            "language": "python",
        })
    
    @contextmanager
    def track_request(self, endpoint: str, method: str = "POST"):
        """追踪 HTTP 请求"""
        start = time.time()
        status = "success"
        try:
            yield self
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start
            REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    
    @contextmanager
    def track_agent_run(self, agent_type: str = "react"):
        """追踪 Agent 执行"""
        start = time.time()
        ACTIVE_AGENTS.inc()
        status = "success"
        steps = 0
        try:
            yield self
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start
            ACTIVE_AGENTS.dec()
            AGENT_RUN_DURATION.labels(agent_type=agent_type, status=status).observe(duration)
            if steps > 0:
                AGENT_STEPS.labels(agent_type=agent_type).observe(steps)
    
    @contextmanager
    def track_llm_call(self, model: str):
        """追踪 LLM 调用"""
        start = time.time()
        try:
            yield self
        except Exception as e:
            LLM_ERRORS.labels(model=model, error_type=type(e).__name__).inc()
            raise
        finally:
            LLM_LATENCY.labels(model=model).observe(time.time() - start)
    
    @contextmanager
    def track_tool_call(self, tool_name: str):
        """追踪工具调用"""
        start = time.time()
        status = "success"
        try:
            yield self
        except Exception:
            status = "error"
            raise
        finally:
            TOOL_CALLS.labels(tool_name=tool_name, status=status).inc()
            TOOL_LATENCY.labels(tool_name=tool_name).observe(time.time() - start)
    
    def record_tokens(self, model: str, input_tokens: int, output_tokens: int):
        """记录 Token 使用量"""
        LLM_TOKENS.labels(model=model, type="input").inc(input_tokens)
        LLM_TOKENS.labels(model=model, type="output").inc(output_tokens)
    
    def record_steps(self, agent_type: str, steps: int):
        """记录推理步数"""
        AGENT_STEPS.labels(agent_type=agent_type).observe(steps)
    
    def get_prometheus_metrics(self) -> bytes:
        """获取 Prometheus 格式的指标"""
        return generate_latest()


# 全局实例
metrics = MetricsCollector()


def timed(metric: Histogram, labels: Dict[str, str] = None):
    """装饰器：自动记录函数执行时间"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


# ============ 演示 ============

def demo():
    """演示指标收集"""
    print("=" * 50)
    print("Agent Metrics Demo")
    print("=" * 50)
    
    collector = MetricsCollector()
    
    # 模拟 Agent 执行
    with collector.track_agent_run("react"):
        print("  Agent 开始执行...")
        
        # LLM 调用
        with collector.track_llm_call("gpt-4o"):
            time.sleep(0.05)
            collector.record_tokens("gpt-4o", 150, 80)
            print("  LLM 调用完成")
        
        # 工具调用
        with collector.track_tool_call("search_stock"):
            time.sleep(0.02)
            print("  工具调用完成")
        
        # 最终 LLM
        with collector.track_llm_call("gpt-4o"):
            time.sleep(0.05)
            collector.record_tokens("gpt-4o", 300, 200)
            print("  最终生成完成")
        
        collector.record_steps("react", 4)
    
    print("\n  指标输出 (Prometheus 格式):")
    print("-" * 40)
    print(collector.get_prometheus_metrics().decode()[:800])
    print("...")
    print("\n✅ 查看: http://localhost:9090 (Prometheus)")
    print("   查看: http://localhost:3000 (Grafana)")


if __name__ == "__main__":
    demo()
