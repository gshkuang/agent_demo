# Agent记忆系统

基于 LangChain Memory + Chroma 的分层记忆架构。

## 架构

```
LangChainMemoryManager
├── ConversationBufferWindowMemory  (短期，窗口=10)
├── ConversationSummaryMemory       (摘要压缩)
├── VectorStoreRetrieverMemory      (长期，Chroma向量检索)
└── SQLite                          (结构化: events/goals/reflections)
```

## 运行

```bash
python3 memory_system.py
```

## 面试Q&A

**Q: 记忆系统怎么设计？**

A: 短期用窗口记忆，长期用向量检索，结构化数据存SQLite。LangChain的ConversationBufferWindowMemory + Chroma向量库直接复用，不需要自己实现。
