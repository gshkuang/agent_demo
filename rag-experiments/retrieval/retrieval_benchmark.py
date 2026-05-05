#!/usr/bin/env python3
"""
检索策略对比 - LangChain版本
使用 Chroma + EnsembleRetriever (BM25 + 向量)
"""
import json, os, time
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


def load_docs(path: str) -> list:
    """加载文档"""
    docs = []
    p = Path(path)
    for f in p.glob("*.md"):
        docs.append({"id": f.stem, "text": f.read_text(encoding="utf-8")})
    json_dir = p / "mx_data" / "output"
    if json_dir.exists():
        for f in list(json_dir.glob("*_raw.json"))[:20]:
            docs.append({"id": f.stem, "text": json.dumps(json.load(f.open()), ensure_ascii=False)})
    return docs


def benchmark_strategy(name: str, retriever, queries: list):
    """测试检索策略"""
    recall_scores = []
    mrr_scores = []
    latencies = []
    
    for q in queries:
        start = time.time()
        results = retriever.invoke(q["query"])
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        ids = [r.metadata.get("id", "") for r in results]
        # 检查是否包含预期文档
        recall = 1.0 if any(r in ids[:5] for r in q["relevant"]) else 0.0
        recall_scores.append(recall)
        
        # MRR
        for i, doc_id in enumerate(ids):
            if doc_id in q["relevant"]:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0.0)
    
    return {
        "strategy": name,
        "recall@5": sum(recall_scores) / len(recall_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
        "avg_latency_ms": sum(latencies) / len(latencies),
    }


def demo():
    print("=" * 50)
    print("Retrieval Benchmark - LangChain")
    print("=" * 50)
    
    docs = load_docs(os.path.expanduser("~/Desktop/agent_demo"))
    texts = [d["text"] for d in docs]
    metadatas = [{"id": d["id"]} for d in docs]
    print(f"\n加载 {len(docs)} 个文档")
    
    queries = [
        {"query": "沪电股份 ROE", "relevant": ["5stocks-beibeixia-maomao-analysis_20260421"]},
        {"query": "泰豪科技 净利润", "relevant": ["mx_data_泰豪科技_营业收入_净利润_2023年_2024年_2025年_raw"]},
        {"query": "芯瑞达 主力资金", "relevant": ["mx_data_芯瑞达_主力资金流向_近5日_raw"]},
        {"query": "板块轮动 涨停", "relevant": ["sector_rotation_20260418_review"]},
        {"query": "贝贝虾 评分", "relevant": ["5stocks-beibeixia-maomao-analysis_20260421"]},
    ]
    
    # 构建检索器
    try:
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
        vector_store = Chroma.from_texts(texts, embedder, metadatas=metadatas)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"向量检索器失败: {e}")
        vector_retriever = None
    
    # BM25 (使用langchain_community)
    from langchain_core.documents import Document
    bm25_docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=5)
    
    strategies = [("bm25", bm25_retriever)]
    if vector_retriever:
        strategies.append(("vector", vector_retriever))
        strategies.append(("hybrid", EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )))
    
    results = []
    for name, retriever in strategies:
        print(f"\n测试 {name}...")
        r = benchmark_strategy(name, retriever, queries)
        results.append(r)
        print(f"  Recall@5: {r['recall@5']:.2f}, MRR: {r['mrr']:.3f}, 延迟: {r['avg_latency_ms']:.1f}ms")
    
    # 保存
    out = Path(os.path.expanduser("~/Desktop/rag-experiments/retrieval/results"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n结果保存: {out}/benchmark_results.json")


if __name__ == "__main__":
    demo()
