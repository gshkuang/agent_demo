# Evaluation 模块

基于 LangSmith + CriteriaEvalChain 的三维评估体系。

## 维度

| 维度 | 指标 | 工具 |
|------|------|------|
| 效能 | 完成率、步数、Token成本 | 自定义统计 |
| 质量 | 准确率、满意度 | CriteriaEvalChain |
| 鲁棒性 | 异常处理、自修复率 | 自定义统计 |

## 运行

```bash
python3 agent_evaluator.py
```

## 面试Q&A

**Q: 如何量化评估Agent？**

A: 三维看板 + LangSmith自动追踪。每次失败归因到意图/规划/工具/记忆，驱动迭代。
