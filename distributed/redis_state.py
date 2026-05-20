#!/usr/bin/env python3
"""
Redis 状态共享 - 多 Agent 分布式状态管理
P3: 基于 Redis 的 Agent 状态同步

功能:
- Agent 状态持久化
- 跨进程/跨机器状态共享
- 分布式锁
- 会话管理
- 实时状态订阅
"""
import os
import json
import time
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import contextmanager

try:
    import redis
    from redis.lock import Lock
except ImportError:
    print("请先安装 Redis: pip install redis")
    raise


# ============ 配置 ============

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class RedisStateManager:
    """Redis 状态管理器 - Agent 分布式状态"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or REDIS_URL
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        self._check_connection()
    
    def _check_connection(self):
        """检查连接"""
        try:
            self.client.ping()
            print(f"✅ Redis 连接成功: {self.redis_url}")
        except Exception as e:
            print(f"⚠️ Redis 连接失败: {e}")
            raise
    
    # ============ Agent 状态 ============
    
    def save_agent_state(self, session_id: str, state: Dict[str, Any], ttl: int = 3600):
        """
        保存 Agent 状态
        
        Args:
            session_id: 会话ID
            state: Agent 状态字典
            ttl: 过期时间（秒）
        """
        key = f"agent:state:{session_id}"
        self.client.setex(key, ttl, json.dumps(state))
    
    def get_agent_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取 Agent 状态"""
        key = f"agent:state:{session_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None
    
    def delete_agent_state(self, session_id: str):
        """删除 Agent 状态"""
        key = f"agent:state:{session_id}"
        self.client.delete(key)
    
    # ============ 会话管理 ============
    
    def create_session(self, user_id: str = None) -> str:
        """创建新会话"""
        session_id = hashlib.md5(f"{user_id or 'anon'}{time.time()}".encode()).hexdigest()[:16]
        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            "message_count": 0,
        }
        self.client.setex(f"session:{session_id}", 86400, json.dumps(session_data))
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        data = self.client.get(f"session:{session_id}")
        return json.loads(data) if data else None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """更新会话信息"""
        session = self.get_session(session_id)
        if session:
            session.update(updates)
            session["last_active"] = datetime.utcnow().isoformat()
            self.client.setex(f"session:{session_id}", 86400, json.dumps(session))
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话历史"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.client.lpush(f"session:{session_id}:messages", json.dumps(message))
        self.client.ltrim(f"session:{session_id}:messages", 0, 99)  # 保留最近100条
        self.update_session(session_id, {"message_count": self.client.llen(f"session:{session_id}:messages")})
    
    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取会话消息历史"""
        messages = self.client.lrange(f"session:{session_id}:messages", 0, limit - 1)
        return [json.loads(m) for m in messages]
    
    # ============ 分布式锁 ============
    
    @contextmanager
    def lock(self, lock_name: str, timeout: int = 30, blocking_timeout: int = 5):
        """
        分布式锁上下文管理器
        
        用法:
            with state_manager.lock("agent:run:session_123"):
                # 执行需要互斥的操作
                pass
        """
        lock = self.client.lock(f"lock:{lock_name}", timeout=timeout, blocking_timeout=blocking_timeout)
        try:
            lock.acquire()
            yield lock
        finally:
            lock.release()
    
    # ============ 任务队列 ============
    
    def enqueue_task(self, queue_name: str, task_data: Dict[str, Any]) -> str:
        """将任务加入队列"""
        task_id = hashlib.md5(f"{queue_name}{time.time()}".encode()).hexdigest()[:16]
        task_data["task_id"] = task_id
        task_data["created_at"] = datetime.utcnow().isoformat()
        self.client.lpush(f"queue:{queue_name}", json.dumps(task_data))
        return task_id
    
    def dequeue_task(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """从队列取出任务"""
        result = self.client.brpop(f"queue:{queue_name}", timeout=5)
        if result:
            return json.loads(result[1])
        return None
    
    def get_queue_length(self, queue_name: str) -> int:
        """获取队列长度"""
        return self.client.llen(f"queue:{queue_name}")
    
    # ============ 发布订阅 ============
    
    def publish(self, channel: str, message: Dict[str, Any]):
        """发布消息到频道"""
        self.client.publish(channel, json.dumps(message))
    
    def subscribe(self, channel: str):
        """订阅频道"""
        pubsub = self.client.pubsub()
        pubsub.subscribe(channel)
        return pubsub
    
    # ============ 统计 ============
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.client.info()
        return {
            "redis_version": info.get("redis_version"),
            "connected_clients": info.get("connected_clients"),
            "used_memory_human": info.get("used_memory_human"),
            "total_keys": self.client.dbsize(),
            "agent_states": len(self.client.keys("agent:state:*")),
            "sessions": len(self.client.keys("session:*")),
        }


# ============ Agent 状态机 ============

class DistributedAgentStateMachine:
    """分布式 Agent 状态机"""
    
    def __init__(self, session_id: str, state_manager: RedisStateManager = None):
        self.session_id = session_id
        self.state = state_manager or RedisStateManager()
        self.state_key = f"agent:state:{session_id}"
    
    def transition(self, from_state: str, to_state: str, data: Dict[str, Any] = None) -> bool:
        """
        状态转换
        
        使用分布式锁保证原子性
        """
        with self.state.lock(f"state_machine:{self.session_id}"):
            current = self.state.get_agent_state(self.session_id)
            
            if current and current.get("status") != from_state:
                print(f"状态转换失败: 当前 {current.get('status')}，期望 {from_state}")
                return False
            
            new_state = {
                "status": to_state,
                "previous_status": from_state,
                "transition_at": datetime.utcnow().isoformat(),
                "data": data or {},
            }
            
            self.state.save_agent_state(self.session_id, new_state)
            print(f"状态转换: {from_state} -> {to_state}")
            return True
    
    def get_status(self) -> str:
        """获取当前状态"""
        state = self.state.get_agent_state(self.session_id)
        return state.get("status", "unknown") if state else "not_found"


# ============ 演示 ============

def demo_state_management():
    """演示状态管理"""
    print("=" * 60)
    print("Redis 状态管理演示")
    print("=" * 60)
    
    try:
        state = RedisStateManager()
    except Exception:
        print("⚠️ Redis 未启动，跳过演示")
        print("   启动命令: docker run -d -p 6379:6379 redis:7-alpine")
        return
    
    # 创建会话
    session_id = state.create_session(user_id="ethon")
    print(f"\n✅ 会话创建: {session_id}")
    
    # 添加消息
    state.add_message(session_id, "user", "分析贵州茅台")
    state.add_message(session_id, "assistant", "正在分析...")
    print(f"   消息数: {state.get_session(session_id)['message_count']}")
    
    # 保存 Agent 状态
    state.save_agent_state(session_id, {
        "status": "reasoning",
        "current_step": 2,
        "tools_used": ["search_stock"],
    })
    print(f"   Agent 状态: {state.get_agent_state(session_id)}")
    
    # 队列操作
    task_id = state.enqueue_task("agent_tasks", {"type": "react", "query": "分析茅台"})
    print(f"\n📤 任务入队: {task_id}")
    print(f"   队列长度: {state.get_queue_length('agent_tasks')}")
    
    task = state.dequeue_task("agent_tasks")
    print(f"   取出任务: {task['task_id'] if task else 'None'}")
    
    # 统计
    print(f"\n📊 Redis 统计:")
    stats = state.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")


def demo_state_machine():
    """演示状态机"""
    print("\n" + "=" * 60)
    print("分布式 Agent 状态机演示")
    print("=" * 60)
    
    try:
        state = RedisStateManager()
    except Exception:
        print("⚠️ Redis 未启动，跳过演示")
        return
    
    session_id = f"demo_{int(time.time())}"
    sm = DistributedAgentStateMachine(session_id, state)
    
    # 状态转换
    print(f"\n会话: {session_id}")
    sm.transition("idle", "running", {"query": "分析茅台"})
    sm.transition("running", "completed", {"result": "买入建议"})
    
    print(f"最终状态: {sm.get_status()}")


if __name__ == "__main__":
    demo_state_management()
    demo_state_machine()
