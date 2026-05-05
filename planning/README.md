# Planning 模块

基于 LangGraph 的 ReAct / Reflexion Agent。

## 文件

| 文件 | 说明 | 核心LangGraph组件 |
|------|------|-------------------|
| `react_agent.py` | ReAct循环 | StateGraph + ToolNode + bind_tools |
| `reflexion_agent.py` | 反思改进 | ConditionalEdge 循环 |

## 运行

```bash
python3 react_agent.py
python3 reflexion_agent.py
```

## 面试Q&A

**Q: ReAct vs Plan-and-Execute怎么选？**

A: 流程固定的用Plan（省成本），开放任务用ReAct（灵活）。Plan的每个步骤里加ReAct式检查点。

**Q: Tree of Thoughts在线上能用吗？**

A: 重度ToT不能在线上用，但轻量化版本是杀手锏。同时生成3条回复，用轻量评判模型选最优。成本3倍，质量提升一档。
