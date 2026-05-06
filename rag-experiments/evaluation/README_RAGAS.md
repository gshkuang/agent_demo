# RAGAS 评测实验

## 概述

使用 [RAGAS](https://docs.ragas.io/) (Retrieval-Augmented Generation Assessment) 框架对 RAG 系统进行专业评测。

## RAGAS 核心指标

| 指标 | 说明 | 评估目标 |
|------|------|---------|
| **faithfulness** | 忠实度 | 回答是否基于检索上下文，无幻觉 |
| **answer_relevancy** | 回答相关性 | 回答是否切题、有用 |
| **context_precision** | 上下文精确率 | 检索到的上下文中有多少是相关的 |
| **context_recall** | 上下文召回率 | 相关上下文被检索到的比例 |
| **context_entity_recall** | 实体召回率 | 关键实体在上下文中的覆盖度 |
| **answer_similarity** | 回答相似度 | 与标准答案的语义相似度 |
| **answer_correctness** | 回答正确性 | 事实正确性综合评分 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `ragas_evaluator.py` | 基础RAGAS评测器，单配置评估 |
| `ragas_experiment.py` | 对比实验，多配置矩阵评测 |
| `evaluation_framework.py` | 原有LangChain评估框架 |

## 快速开始

### 1. 安装依赖

```bash
pip install ragas
```

### 2. 基础评测

```bash
cd ~/Desktop/agent_demo/rag-experiments/evaluation
python3 ragas_evaluator.py
```

### 3. 对比实验

```bash
python3 ragas_experiment.py
```

## 实验配置矩阵

| 配置 | 分块策略 | 嵌入模型 | 检索策略 |
|------|---------|---------|---------|
| 1 | fixed | bge | vector |
| 2 | recursive | bge | vector |
| 3 | recursive | openai | vector |
| 4 | recursive | bge | bm25 |
| 5 | recursive | bge | hybrid |
| 6 | markdown | bge | vector |

## 与 LangChain 评估对比

| 特性 | RAGAS | LangChain CriteriaEval |
|------|-------|----------------------|
| 指标维度 | 7个细粒度指标 | 2-3个自定义标准 |
| 自动化 | 全自动LLM评判 | 半自动，需配置 |
| 上下文分析 | 精确率/召回率拆解 | 无 |
| 实体分析 | 支持 | 不支持 |
| 依赖 | 较重 | 轻量 |
| 集成 | 独立框架 | LangSmith追踪 |

## 使用建议

- **开发调优阶段**: 使用 RAGAS 进行深度分析，定位问题环节
- **生产监控阶段**: 使用 LangChain + LangSmith 持续追踪
- **两者互补**: RAGAS 用于深度诊断，LangChain 用于日常监控

## 输出示例

```
📊 RAGAS 评测报告
============================================================
总样本数: 5
平均评测耗时: 2450ms

【核心指标】
  faithfulness        : 0.920 (范围: 0.850-1.000, n=5)
  answer_relevancy    : 0.880 (范围: 0.800-0.950, n=5)
  context_precision   : 0.750 (范围: 0.600-0.900, n=5)
  context_recall      : 0.820 (范围: 0.700-0.950, n=5)

【综合评分】0.843 / 1.0
  ✅ 优秀 - RAG系统表现良好
```

## 面试Q&A

**Q: 如何系统评估RAG系统？**

A: 三维评估体系：
1. **检索质量**: Recall@K, Precision@K, MRR (传统IR指标)
2. **生成质量**: RAGAS指标 (faithfulness, relevancy等)
3. **端到端**: 任务完成率、用户满意度

RAGAS的优势在于自动化和细粒度，能定位是检索问题还是生成问题。

**Q: RAGAS的faithfulness怎么计算？**

A: 两个步骤：
1. 将回答拆分为独立陈述句
2. 用NLI模型判断每个陈述是否能从上下文中推断
3. 忠实度 = 可推断陈述数 / 总陈述数

**Q: 上下文精确率和召回率的区别？**

A: 
- Precision: 检索到的K个文档中，有多少是相关的
- Recall: 所有相关文档中，有多少被检索到了
- 精确率低 → 检索噪音大，需要优化排序
- 召回率低 → 漏掉了相关文档，需要扩大检索范围
