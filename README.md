# agent_demo - AI Agent实验项目

基于 LangChain/LangGraph 的Agent实验集合，不再重复造轮子。

## 项目结构

```
agent_demo/
├── planning/              # ReAct / Reflexion Agent (LangGraph)
├── memory/                # 记忆系统 (LangChain Memory + Chroma)
├── tool_safety/           # 工具安全 (BaseTool包装 + Pydantic校验)
├── multi_agent/           # Manager-Worker (LangGraph Send)
├── evaluation/            # 三维评估 (LangSmith + CriteriaEvalChain)
├── rag-experiments/       # RAG实验 (保持独立，不复用)
└── data/                  # 数据目录
```

## 技术栈

- **LangGraph**: Agent工作流编排 (ReAct循环、多Agent分发、反思)
- **LangChain**: 工具包装、Memory管理、评估器
- **LangSmith**: 运行追踪与评估
- **Chroma**: 向量存储
- **OpenAI**: LLM + Embedding

## 快速开始

```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'

# 各模块演示
cd planning && python3 react_agent.py
cd planning && python3 reflexion_agent.py
cd multi_agent && python3 manager_worker.py
cd memory && python3 memory_system.py
cd evaluation && python3 agent_evaluator.py
cd tool_safety && python3 safe_tool_executor.py
```

## 面试考点覆盖

| 面试题 | 模块 | 文件 |
|--------|------|------|
| ReAct vs Plan-and-Execute | planning | react_agent.py |
| 工具安全三层防护 | tool_safety | safe_tool_executor.py |
| 多Agent协作 | multi_agent | manager_worker.py |
| 记忆系统设计 | memory | memory_system.py |
| Agent三维评估 | evaluation | agent_evaluator.py |
| RAG系统设计 | rag-experiments | 独立实验 |

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
