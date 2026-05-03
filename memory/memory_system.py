#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent记忆系统 - 分层架构实现

核心设计:
1. 短期记忆: 当前会话上下文（内存）
2. 长期记忆: 向量存储 + 结构化存储（SQLite）
3. 元记忆: 反思日志与执行模式

参考: https://developer.aliyun.com/article/1714493
"""

import json
import sqlite3
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime


# ============== 数据模型 ==============

@dataclass
class MemoryEvent:
    """记忆事件"""
    id: str
    timestamp: float
    agent_name: str
    input_text: str
    output_text: str
    summary: str
    embedding: Optional[List[float]] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Goal:
    """长期目标"""
    id: str
    agent_name: str
    goal_text: str
    status: str = "in_progress"  # in_progress / completed / failed
    created_at: float = None
    last_updated: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_updated is None:
            self.last_updated = self.created_at


# ============== 短期记忆 ==============

class ShortTermMemory:
    """
    短期记忆 - 工作记忆
    保持当前会话的交互上下文
    """
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history: List[Dict] = []
        self.context_summary: str = ""
    
    def add_turn(self, user_input: str, agent_output: str):
        """添加一轮对话"""
        self.conversation_history.append({
            "timestamp": time.time(),
            "user": user_input,
            "agent": agent_output
        })
        
        # 保持最近N轮
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]
    
    def get_context(self, n_turns: int = None) -> str:
        """获取格式化上下文"""
        turns = self.conversation_history[-n_turns:] if n_turns else self.conversation_history
        context = []
        for turn in turns:
            context.append(f"User: {turn['user']}")
            context.append(f"Agent: {turn['agent']}")
        return "\n".join(context)
    
    def clear(self):
        """清空短期记忆"""
        self.conversation_history = []
        self.context_summary = ""


# ============== 长期记忆 - 向量存储 ==============

class VectorMemoryStore:
    """
    向量记忆存储
    使用BGE模型生成语义向量，支持相似度检索
    """
    
    def __init__(self, model_path: str = "/tmp/models/BAAI/bge-large-zh-v1___5"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_path)
            self.available = True
        except Exception as e:
            print(f"⚠️ BGE模型加载失败: {e}")
            self.available = False
            self.model = None
        
        self.memories: Dict[str, Dict] = {}  # id -> {text, embedding, metadata}
    
    def add(self, text: str, metadata: Dict = None) -> str:
        """添加记忆"""
        mem_id = hashlib.md5(f"{text}_{time.time()}".encode()).hexdigest()[:12]
        
        if self.available:
            embedding = self.model.encode(text, normalize_embeddings=True).tolist()
        else:
            # Fallback: 随机向量（仅测试用）
            embedding = np.random.randn(1024).tolist()
        
        self.memories[mem_id] = {
            "id": mem_id,
            "text": text,
            "embedding": embedding,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        return mem_id
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义检索"""
        if not self.memories:
            return []
        
        if self.available:
            query_vec = self.model.encode(query, normalize_embeddings=True)
        else:
            query_vec = np.random.randn(1024)
        
        # 计算相似度
        scores = []
        for mem_id, mem in self.memories.items():
            mem_vec = np.array(mem["embedding"])
            similarity = np.dot(query_vec, mem_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)
            )
            scores.append((mem_id, float(similarity)))
        
        # 排序返回
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for mem_id, score in scores[:top_k]:
            mem = self.memories[mem_id].copy()
            mem["score"] = score
            results.append(mem)
        
        return results
    
    def get_all(self) -> List[Dict]:
        """获取所有记忆"""
        return list(self.memories.values())


# ============== 长期记忆 - 结构化存储 ==============

class StructuredMemoryStore:
    """
    结构化记忆存储
    使用SQLite存储事件、目标、偏好等
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / "Desktop" / "agent_demo" / "memory" / "agent_memory.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 记忆事件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                agent_name TEXT,
                input_text TEXT,
                output_text TEXT,
                summary TEXT,
                embedding BLOB,
                metadata TEXT
            )
        """)
        
        # 目标表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                agent_name TEXT,
                goal_text TEXT,
                status TEXT,
                created_at REAL,
                last_updated REAL
            )
        """)
        
        # 用户偏好表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id TEXT PRIMARY KEY,
                agent_name TEXT,
                key TEXT,
                value TEXT,
                updated_at REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_event(self, event: MemoryEvent):
        """保存记忆事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = json.dumps(event.embedding) if event.embedding else None
        metadata_json = json.dumps(event.metadata, ensure_ascii=False)
        
        cursor.execute("""
            INSERT OR REPLACE INTO memory_events 
            (id, timestamp, agent_name, input_text, output_text, summary, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.id, event.timestamp, event.agent_name,
            event.input_text, event.output_text, event.summary,
            embedding_blob, metadata_json
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_events(self, agent_name: str, limit: int = 10) -> List[MemoryEvent]:
        """获取近期事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM memory_events 
            WHERE agent_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (agent_name, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            embedding = json.loads(row[6]) if row[6] else None
            metadata = json.loads(row[7]) if row[7] else {}
            events.append(MemoryEvent(
                id=row[0], timestamp=row[1], agent_name=row[2],
                input_text=row[3], output_text=row[4], summary=row[5],
                embedding=embedding, metadata=metadata
            ))
        
        return events
    
    def save_goal(self, goal: Goal):
        """保存目标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO goals 
            (id, agent_name, goal_text, status, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (goal.id, goal.agent_name, goal.goal_text, goal.status,
              goal.created_at, goal.last_updated))
        
        conn.commit()
        conn.close()
    
    def get_active_goals(self, agent_name: str) -> List[Goal]:
        """获取进行中的目标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM goals 
            WHERE agent_name = ? AND status = 'in_progress'
            ORDER BY created_at DESC
        """, (agent_name,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Goal(id=r[0], agent_name=r[1], goal_text=r[2], 
                     status=r[3], created_at=r[4], last_updated=r[5]) 
                for r in rows]


# ============== 元记忆 ==============

class MetaMemory:
    """
    元记忆 - 关于记忆的记忆
    记录反思、模式识别、学习经验
    """
    
    def __init__(self):
        self.reflections: List[Dict] = []
        self.patterns: Dict[str, int] = {}  # 模式 -> 出现次数
    
    def add_reflection(self, task: str, outcome: str, lesson: str):
        """添加反思记录"""
        self.reflections.append({
            "timestamp": time.time(),
            "task": task,
            "outcome": outcome,
            "lesson": lesson
        })
    
    def record_pattern(self, pattern_type: str, description: str):
        """记录行为模式"""
        key = f"{pattern_type}:{description}"
        self.patterns[key] = self.patterns.get(key, 0) + 1
    
    def get_insights(self) -> List[str]:
        """获取洞察（高频模式）"""
        sorted_patterns = sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)
        return [f"{k} (出现{v}次)" for k, v in sorted_patterns[:5]]


# ============== 统一记忆管理器 ==============

class AgentMemoryManager:
    """
    Agent统一记忆管理器
    整合短期、长期、元记忆三层架构
    """
    
    def __init__(self, agent_name: str = "default_agent"):
        self.agent_name = agent_name
        
        # 三层记忆
        self.short_term = ShortTermMemory(max_turns=10)
        self.vector_store = VectorMemoryStore()
        self.structured_store = StructuredMemoryStore()
        self.meta_memory = MetaMemory()
        
        print(f"✅ AgentMemoryManager初始化完成 [{agent_name}]")
    
    def process_interaction(self, user_input: str, agent_output: str, 
                           summary: str = None):
        """
        处理一次交互，更新所有记忆层
        """
        # 1. 更新短期记忆
        self.short_term.add_turn(user_input, agent_output)
        
        # 2. 生成摘要（简化版，实际可用LLM）
        if summary is None:
            summary = f"用户询问: {user_input[:50]}... -> Agent回答: {agent_output[:50]}..."
        
        # 3. 保存到向量存储
        mem_text = f"Q: {user_input}\nA: {agent_output}"
        mem_id = self.vector_store.add(mem_text, {
            "agent": self.agent_name,
            "type": "interaction"
        })
        
        # 4. 保存到结构化存储
        event = MemoryEvent(
            id=mem_id,
            timestamp=time.time(),
            agent_name=self.agent_name,
            input_text=user_input,
            output_text=agent_output,
            summary=summary,
            metadata={"type": "interaction"}
        )
        self.structured_store.save_event(event)
        
        return mem_id
    
    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        回忆相关记忆（向量检索 + 结构化过滤）
        """
        # 向量检索
        vector_results = self.vector_store.search(query, top_k)
        
        # 补充近期事件
        recent_events = self.structured_store.get_recent_events(self.agent_name, limit=3)
        
        # 合并结果
        combined = {
            "vector_results": vector_results,
            "recent_context": self.short_term.get_context(3),
            "recent_events": [
                {"summary": e.summary, "timestamp": e.timestamp} 
                for e in recent_events
            ]
        }
        
        return combined
    
    def set_goal(self, goal_text: str) -> str:
        """设置长期目标"""
        goal_id = hashlib.md5(f"{goal_text}_{time.time()}".encode()).hexdigest()[:12]
        goal = Goal(id=goal_id, agent_name=self.agent_name, goal_text=goal_text)
        self.structured_store.save_goal(goal)
        return goal_id
    
    def get_active_goals(self) -> List[Goal]:
        """获取进行中的目标"""
        return self.structured_store.get_active_goals(self.agent_name)
    
    def reflect(self, task: str, outcome: str, lesson: str):
        """记录反思"""
        self.meta_memory.add_reflection(task, outcome, lesson)
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            "short_term_turns": len(self.short_term.conversation_history),
            "vector_memories": len(self.vector_store.memories),
            "active_goals": len(self.get_active_goals()),
            "reflections": len(self.meta_memory.reflections),
            "patterns": len(self.meta_memory.patterns)
        }


# ============== 测试 ==============

def test_memory_system():
    """测试记忆系统"""
    print("=" * 60)
    print("Agent记忆系统测试")
    print("=" * 60)
    
    # 初始化
    memory = AgentMemoryManager(agent_name="ethon")
    
    # 模拟交互
    print("\n📝 记录交互...")
    memory.process_interaction(
        "分析沪电股份的ROE",
        "沪电股份2024年ROE为18.5%，高于行业平均水平...",
        "财务分析: 沪电股份ROE分析"
    )
    memory.process_interaction(
        "查看三安光电的股东信息",
        "三安光电十大流通股东包括福建三安集团、香港中央结算有限公司...",
        "股东分析: 三安光电十大流通股东"
    )
    memory.process_interaction(
        "板块轮动复盘",
        "今日板块轮动路径: 半导体 -> 新能源 -> 医药...",
        "板块分析: 2026-05-03板块轮动复盘"
    )
    
    # 测试回忆
    print("\n🔍 测试回忆功能...")
    results = memory.recall("沪电股份财务数据", top_k=3)
    print(f"  向量检索结果: {len(results['vector_results'])}条")
    for r in results['vector_results']:
        print(f"    - {r['text'][:60]}... (score: {r['score']:.3f})")
    
    # 测试目标
    print("\n🎯 测试目标管理...")
    goal_id = memory.set_goal("完成本周投资组合分析")
    print(f"  设置目标: {goal_id}")
    
    goals = memory.get_active_goals()
    print(f"  进行中目标: {len(goals)}个")
    
    # 测试反思
    print("\n💭 测试反思功能...")
    memory.reflect("ROE分析", "成功", "使用杜邦分析法更有效")
    print(f"  反思记录数: {len(memory.meta_memory.reflections)}")
    
    # 统计
    print("\n📊 记忆统计:")
    stats = memory.get_memory_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    test_memory_system()
