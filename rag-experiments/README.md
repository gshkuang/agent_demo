# RAG Experiments - 金融文档RAG系统实验集合

## 项目概述

本项目是一系列针对金融文档场景的RAG（检索增强生成）系统实验，涵盖从文本分块、向量嵌入、检索策略到端到端评估的完整链路。

## 实验模块

| 模块 | 状态 | 说明 | 核心结论 |
|------|:----:|------|---------|
| [chunking](./chunking/) | ✅ 完成 | 文本分块策略对比 | 递归分块和结构化感知分块最佳 |
| [embedding](./embedding/) | ✅ 完成 | 向量嵌入模型对比 | m3e-base性价比最高，bge-large-zh精度最好 |
| [retrieval](./retrieval/) | ✅ 完成 | 检索策略优化 | BM25在关键词匹配上优于纯向量检索 |
| [evaluation](./evaluation/) | ✅ 完成 | 端到端评估框架 | 结构化感知+递归分块+混合检索综合最优 |

## 快速开始

```bash
# 克隆项目
git clone <repository-url>
cd rag-experiments

# 运行各模块实验
cd chunking && python3 rag_chunking_test_v2.py
cd ../embedding && python3 embedding_benchmark.py
cd ../retrieval && python3 retrieval_benchmark.py
cd ../evaluation && python3 evaluation_framework.py
```

## 项目结构

```
rag-experiments/
├── README.md              # 本文件
├── chunking/             # ✅ 文本分块实验
│   ├── README.md
│   ├── rag_chunking_test_v2.py
│   ├── rag_chunking_test.py
│   ├── RAG_CHUNKING_TEST_REPORT.md
│   └── chunking_test_results_v2.json
├── embedding/            # ✅ 向量嵌入实验
│   ├── README.md
│   ├── embedding_benchmark.py
│   └── results/
│       └── benchmark_results.json
├── retrieval/            # ✅ 检索策略实验
│   ├── README.md
│   ├── retrieval_benchmark.py
│   └── results/
│       └── benchmark_results.json
├── evaluation/           # ✅ 端到端评估实验
│   ├── README.md
│   ├── evaluation_framework.py
│   └── results/
│       └── e2e_evaluation.json
└── docs/                 # 文档资料
    └── references/       # 参考论文/文章
```

## 核心发现汇总

### 1. Chunking实验

**最佳策略**: 递归分块、结构化感知分块（并列）

| 策略 | 完全命中 | 边界质量 |
|------|:--------:|:--------:|
| 递归分块 | 90.0% | 100% |
| 结构化感知分块 | 90.0% | 100% |
| 固定长度分块 | 90.0% | 74% |

### 2. Embedding实验

| 模型 | 维度 | Recall@5 | MRR | 延迟 |
|------|:----:|:--------:|:---:|:----:|
| text-embedding-3-small | 1536 | 6.2% | 0.042 | 21.7ms |
| bge-large-zh | 1024 | 10.4% | 0.185 | 26.1ms |
| m3e-base | 768 | 10.4% | 0.203 | 17.3ms |

**结论**: m3e-base在精度和速度上平衡最好，bge-large-zh在MRR上领先

### 3. Retrieval实验

| 策略 | Recall@1 | Recall@5 | MRR |
|------|:--------:|:--------:|:---:|
| 纯向量检索 | 0.0% | 25.0% | 0.094 |
| 纯BM25检索 | 25.0% | 50.0% | 0.354 |
| 混合检索(RRF) | 12.5% | 12.5% | 0.125 |
| 混合检索(线性α=0.5) | 12.5% | 25.0% | 0.167 |

**结论**: BM25在金融关键词检索上明显优于纯向量检索

### 4. Evaluation实验

**最佳配置**: structured + text-embedding-3-small + hybrid_rrf

| 指标 | 得分 |
|------|:----:|
| 检索Recall | 100% |
| 忠实度 | 100% |
| 完整性 | 85% |

## 技术栈

- **Python 3.9+**
- **向量数据库**: ChromaDB / Milvus（计划中）
- **嵌入模型**: OpenAI / BGE / M3E
- **LLM**: GPT-4 / Claude / 本地模型

## 参考资源

- [掘金 - 切实有效的RAG文本分块](https://juejin.cn/post/7355666189475954725)
- [LangChain Document Transformers](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LlamaIndex Node Parser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)

## 许可证

MIT License

---

*项目初始化: 2026-05-03*
*所有实验完成: 2026-05-03*
*维护者: Ethon*
