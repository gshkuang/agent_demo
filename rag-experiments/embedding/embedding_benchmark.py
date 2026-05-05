#!/usr/bin/env python3
"""
Embedding模型对比 - LangChain版本
使用 langchain_openai + langchain_huggingface
"""
import json, os, time
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_chunks(path: str) -> list:
    """加载文本块"""
    chunks = []
    p = Path(path)
    for f in p.glob("*.md"):
        text = f.read_text(encoding="utf-8")
        for i, para in enumerate(text.split("\n\n")):
            if para.strip():
                chunks.append({"id": f"{f.stem}_p{i}", "text": para.strip()})
    return chunks


def benchmark_model(name: str, embedder, chunks: list, queries: list):
    """测试单个模型"""
    # 构建向量存储
    store = Chroma.from_texts(
        texts=[c["text"] for c in chunks],
        embedding=embedder,
        metadatas=[{"id": c["id"]} for c in chunks]
    )
    
    # 嵌入耗时
    start = time.time()
    _ = embedder.embed_documents([c["text"] for c in chunks[:10]])
    embed_time = (time.time() - start) * 1000 / 10
    
    # 检索测试
    recall_scores = []
    query_times = []
    
    for q in queries:
        start = time.time()
        results = store.similarity_search(q["query"], k=5)
        query_times.append((time.time() - start) * 1000)
        
        # 简单评估：检查关键词匹配
        found = any(kw in results[0].page_content for kw in q["keywords"]) if results else False
        recall_scores.append(1.0 if found else 0.0)
    
    return {
        "model": name,
        "recall@5": sum(recall_scores) / len(recall_scores),
        "avg_embed_ms": embed_time,
        "avg_query_ms": sum(query_times) / len(query_times),
    }


def demo():
    print("=" * 50)
    print("Embedding Benchmark - LangChain")
    print("=" * 50)
    
    chunks = load_chunks(os.path.expanduser("~/Desktop/agent_demo"))
    print(f"\n加载 {len(chunks)} 个文本块")
    
    queries = [
        {"query": "沪电股份 ROE", "keywords": ["沪电", "ROE"]},
        {"query": "泰豪科技 净利润", "keywords": ["泰豪", "利润"]},
        {"query": "芯瑞达 主力资金", "keywords": ["芯瑞达", "资金"]},
        {"query": "板块轮动 涨停", "keywords": ["轮动", "涨停"]},
    ]
    
    # 测试模型
    models = [
        ("openai", OpenAIEmbeddings(model="text-embedding-3-small")),
        ("bge", HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")),
    ]
    
    results = []
    for name, embedder in models:
        print(f"\n测试 {name}...")
        try:
            r = benchmark_model(name, embedder, chunks, queries)
            results.append(r)
            print(f"  Recall@5: {r['recall@5']:.2f}, 嵌入: {r['avg_embed_ms']:.1f}ms, 查询: {r['avg_query_ms']:.1f}ms")
        except Exception as e:
            print(f"  失败: {e}")
    
    # 保存
    out = Path(os.path.expanduser("~/Desktop/rag-experiments/embedding/results"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n结果保存: {out}/benchmark_results.json")


if __name__ == "__main__":
    demo()
