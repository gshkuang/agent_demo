# Agent记忆系统

基于阿里云文章《AI Agent记忆机制详解》实现的分层记忆架构。

## 架构设计

```
┌─────────────────────────────────────────┐
│         AgentMemoryManager              │
├─────────────────────────────────────────┤
│  短期记忆 (ShortTermMemory)              │
│  ├── 当前会话上下文 (max_turns=10)        │
│  └── 运行时缓存                          │
├─────────────────────────────────────────┤
│  长期记忆                                │
│  ├── VectorMemoryStore (BGE语义检索)     │
│  └── StructuredMemoryStore (SQLite)      │
├─────────────────────────────────────────┤
│  元记忆 (MetaMemory)                     │
│  ├── 反思日志                            │
│  └── 行为模式                            │
└─────────────────────────────────────────┘
```

## 核心特性

| 特性 | 实现 | 状态 |
|------|------|------|
| 语义检索 | BGE-large-zh-v1.5 + 余弦相似度 | ✅ |
| 结构化存储 | SQLite (events/goals/preferences) | ✅ |
| 目标追踪 | 长期目标管理 | ✅ |
| 反思机制 | 元记忆记录 | ✅ |
| 记忆压缩 | 大工具卸载 + 摘要压缩 + 池压缩 | ✅ |
| 遗忘曲线 | 艾宾浩斯衰减 (7天半衰期) | ✅ |
| 知识图谱 | 待实现 | 🔄 |

## 快速开始

```python
from memory.memory_system import AgentMemoryManager

# 初始化
memory = AgentMemoryManager(agent_name="ethon")

# 记录交互
memory.process_interaction(
    user_input="分析沪电股份的ROE",
    agent_output="沪电股份2024年ROE为18.5%...",
    summary="财务分析: 沪电股份ROE"
)

# 回忆相关记忆
results = memory.recall("沪电股份财务数据", top_k=3)
# 返回: {vector_results, recent_context, recent_events}

# 设置目标
memory.set_goal("完成本周投资组合分析")

# 记录反思
memory.reflect("ROE分析", "成功", "使用杜邦分析法更有效")

# 记忆压缩 (上下文超限时)
summary = memory.compress_short_term()

# 向量记忆池压缩 (基于遗忘曲线)
memory.compress_vector_memories(target_size=100)

# 查看统计
print(memory.get_memory_stats())
```

## 数据库Schema

### memory_events表
```sql
CREATE TABLE memory_events (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    agent_name TEXT,
    input_text TEXT,
    output_text TEXT,
    summary TEXT,
    embedding BLOB,
    metadata TEXT
);
```

### goals表
```sql
CREATE TABLE goals (
    id TEXT PRIMARY KEY,
    agent_name TEXT,
    goal_text TEXT,
    status TEXT,
    created_at REAL,
    last_updated REAL
);
```

## 参考

- [AI Agent记忆机制详解-阿里云](https://developer.aliyun.com/article/1714493)
- [Agent记忆机制-知乎](https://zhuanlan.zhihu.com/p/2033633355338657966)
