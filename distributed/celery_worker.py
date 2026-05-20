#!/usr/bin/env python3
"""
分布式 Agent 任务队列 - Celery + Redis
P3: 生产级异步 Agent 执行

功能:
- 异步 Agent 任务提交
- 任务状态追踪
- 分布式 Worker 扩展
- 结果持久化
- 失败重试

架构:
- Redis: 消息队列 + 结果后端
- Celery Worker: Agent 执行器
- Flower: 任务监控
"""
import os
import json
import time
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from celery import Celery, Task
    from celery.result import AsyncResult
    from celery.signals import task_prerun, task_postrun, task_failure
except ImportError:
    print("请先安装 Celery: pip install celery[redis]")
    raise

# ============ 配置 ============

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# 创建 Celery 应用
app = Celery(
    "agent_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["distributed.celery_worker"],
)

# Celery 配置
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5分钟超时
    task_soft_time_limit=240,  # 4分钟软超时
    worker_prefetch_multiplier=1,  # 公平调度
    worker_max_tasks_per_child=50,  # 防止内存泄漏
    result_expires=3600,  # 结果保留1小时
    broker_connection_retry_on_startup=True,
)


# ============ 任务定义 ============

class AgentTask(Task):
    """Agent 任务基类"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败处理"""
        print(f"[Task {task_id}] 失败: {exc}")
        # 可以在这里发送告警通知
    
    def on_success(self, retval, task_id, args, kwargs):
        """任务成功处理"""
        print(f"[Task {task_id}] 成功")


@app.task(base=AgentTask, bind=True, max_retries=3)
def run_react_agent(self, query: str, session_id: str = None, max_steps: int = 10) -> Dict[str, Any]:
    """
    执行 ReAct Agent 任务
    
    Args:
        query: 用户查询
        session_id: 会话ID
        max_steps: 最大推理步数
    
    Returns:
        执行结果
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from planning.react_agent import build_react_agent, TOOLS
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    task_id = self.request.id
    session_id = session_id or hashlib.md5(f"{time.time()}".encode()).hexdigest()[:16]
    
    print(f"[Task {task_id}] 开始执行: {query[:50]}...")
    
    try:
        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"step": "initializing", "progress": 10, "session_id": session_id},
        )
        
        # 初始化 LLM
        llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=0,
        )
        
        self.update_state(state="PROGRESS", meta={"step": "reasoning", "progress": 30})
        
        # 构建并执行 Agent
        agent = build_react_agent(llm, TOOLS)
        result = agent.invoke({
            "messages": [HumanMessage(content=query)],
            "current_step": 0,
            "max_steps": max_steps,
        })
        
        self.update_state(state="PROGRESS", meta={"step": "finalizing", "progress": 90})
        
        # 提取结果
        steps = result["current_step"]
        messages = result["messages"]
        
        # 提取最终回复
        final_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                final_content = msg.content
                break
        
        # 提取工具调用
        tools_used = []
        for msg in messages:
            if getattr(msg, "tool_calls", None):
                tools_used.extend([t["name"] for t in msg.tool_calls])
        
        output = {
            "task_id": task_id,
            "session_id": session_id,
            "query": query,
            "response": final_content,
            "steps": steps,
            "tools_used": list(set(tools_used)),
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        print(f"[Task {task_id}] 完成，步数: {steps}")
        return output
        
    except Exception as exc:
        # 重试逻辑
        if self.request.retries < self.max_retries:
            print(f"[Task {task_id}] 失败，第 {self.request.retries + 1} 次重试...")
            raise self.retry(exc=exc, countdown=5)
        
        # 最终失败
        return {
            "task_id": task_id,
            "session_id": session_id,
            "query": query,
            "response": f"执行失败: {str(exc)}",
            "steps": 0,
            "tools_used": [],
            "status": "error",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.task(base=AgentTask, bind=True)
def run_multi_agent_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行多 Agent 协作任务
    
    Args:
        task_config: 任务配置
            {
                "agents": ["researcher", "analyst", "writer"],
                "task": "分析贵州茅台",
                "workflow": "sequential"
            }
    """
    task_id = self.request.id
    print(f"[Task {task_id}] 多 Agent 任务: {task_config.get('task', '')}")
    
    # TODO: 集成 multi_agent/manager_worker.py
    
    return {
        "task_id": task_id,
        "status": "completed",
        "result": "多 Agent 任务完成",
    }


@app.task(base=AgentTask)
def health_check() -> Dict[str, Any]:
    """健康检查任务"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "worker": "agent-celery-worker",
    }


# ============ 任务状态查询 ============

def get_task_status(task_id: str) -> Dict[str, Any]:
    """查询任务状态"""
    result = AsyncResult(task_id, app=app)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
        "traceback": result.traceback if result.failed() else None,
    }


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """取消任务"""
    app.control.revoke(task_id, terminate=terminate)
    return True


# ============ 信号处理 ============

@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extras):
    """任务开始前的处理"""
    print(f"[Signal] 任务开始: {task_id}")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **extras):
    """任务完成后的处理"""
    print(f"[Signal] 任务完成: {task_id}, 状态: {state}")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **extras):
    """任务失败处理"""
    print(f"[Signal] 任务失败: {task_id}, 错误: {exception}")


# ============ 演示 ============

def demo_submit_task():
    """演示提交任务"""
    print("=" * 60)
    print("Celery Agent 任务队列演示")
    print("=" * 60)
    
    # 提交任务
    query = "分析贵州茅台的投资价值"
    task = run_react_agent.delay(query, max_steps=5)
    
    print(f"\n📤 任务已提交")
    print(f"   Task ID: {task.id}")
    print(f"   查询: {query}")
    
    # 查询状态
    print(f"\n⏳ 等待执行...")
    time.sleep(2)
    
    status = get_task_status(task.id)
    print(f"   状态: {status['status']}")
    
    if status["status"] == "SUCCESS":
        result = status["result"]
        print(f"\n✅ 任务完成")
        print(f"   回复: {result.get('response', '')[:100]}...")
        print(f"   步数: {result.get('steps', 0)}")
        print(f"   工具: {result.get('tools_used', [])}")
    else:
        print(f"\n⏳ 任务仍在执行中，请稍后查询")
        print(f"   查询命令: python -c \"from distributed.celery_worker import get_task_status; print(get_task_status('{task.id}'))\"")


def demo_health_check():
    """演示健康检查"""
    print("\n" + "-" * 40)
    print("健康检查")
    print("-" * 40)
    
    result = health_check.delay()
    print(f"Health Check Task ID: {result.id}")
    
    # 同步等待结果（仅演示）
    try:
        output = result.get(timeout=5)
        print(f"状态: {output['status']}")
        print(f"时间: {output['timestamp']}")
    except Exception as e:
        print(f"获取结果失败: {e}")


if __name__ == "__main__":
    # 启动 Worker 命令:
    # celery -A distributed.celery_worker worker --loglevel=info --concurrency=2
    
    # 启动 Flower 监控:
    # celery -A distributed.celery_worker flower --port=5555
    
    demo_submit_task()
    # demo_health_check()
