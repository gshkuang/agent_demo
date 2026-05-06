# RAG Experiments - LangChain版本

使用LangChain/LangGraph生态重构的RAG实验，代码精简，不复造轮子。

## 实验模块

| 模块 | 核心组件 | 文件 |
|------|---------|------|
| 文本分块 | RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter | chunking/rag_chunking_test_v2.py |
| 向量嵌入 | HuggingFaceEmbeddings, OpenAIEmbeddings, Chroma | embedding/embedding_benchmark.py |
| 检索策略 | Chroma, BM25Retriever, EnsembleRetriever | retrieval/retrieval_benchmark.py |
| 端到端评估 (LangChain) | load_evaluator(Criteria), ChatOpenAI | evaluation/evaluation_framework.py |
| **RAGAS评测** | **faithfulness, context_precision, answer_relevancy** | **evaluation/ragas_evaluator.py** |
| **RAGAS对比实验** | **多配置矩阵评测** | **evaluation/ragas_experiment.py** |

## 快速开始

```bash
pip install -r ../requirements.txt

# 分块实验
cd chunking && python3 rag_chunking_test_v2.py

# Embedding对比
cd ../embedding && python3 embedding_benchmark.py

# 检索策略
cd ../retrieval && python3 retrieval_benchmark.py

# 端到端评估
cd ../evaluation && python3 evaluation_framework.py
```

## 核心结论

| 模块 | 最佳方案 |
|------|---------|
| 文本分块 | RecursiveCharacterTextSplitter |
| 向量嵌入 | bge-large-zh-v1.5 (HuggingFaceEmbeddings) |
| 检索策略 | EnsembleRetriever (BM25 + 向量, weights=[0.5, 0.5]) |
| 评估 (LangChain) | CriteriaEvalChain (faithfulness + relevance) |
| **评估 (RAGAS)** | **7维度专业评测: faithfulness + relevance + precision + recall + entity + similarity + correctness** |

---
*LangChain重构版*
