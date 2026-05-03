#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding模型对比实验
测试不同Embedding模型在金融文档上的向量表示效果
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict


# ============== 数据模型 ==============

@dataclass
class EmbeddingResult:
    """单个文档的嵌入结果"""
    doc_id: str
    text: str
    vector: List[float]
    model_name: str
    dimension: int
    latency_ms: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    top_k_docs: List[Tuple[str, float]]  # (doc_id, score)
    latency_ms: float

@dataclass
class ModelScore:
    """模型评分"""
    model_name: str
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    avg_latency_ms: float
    avg_similarity: float


# ============== 模拟Embedding模型 ==============

class MockEmbeddingModel:
    """
    模拟Embedding模型行为
    实际生产环境应替换为真实模型API
    """
    
    MODELS = {
        "text-embedding-3-small": {"dim": 1536, "speed": 50},
        "text-embedding-3-large": {"dim": 3072, "speed": 30},
        "bge-large-zh": {"dim": 1024, "speed": 40},
        "m3e-base": {"dim": 768, "speed": 60},
        "bce-embedding-base": {"dim": 768, "speed": 55},
    }
    
    def __init__(self, model_name: str):
        if model_name not in self.MODELS:
            raise ValueError(f"未知模型: {model_name}")
        self.model_name = model_name
        self.dim = self.MODELS[model_name]["dim"]
        self.speed = self.MODELS[model_name]["speed"]
    
    def embed(self, text: str) -> Tuple[List[float], float]:
        """
        生成模拟嵌入向量
        使用文本哈希确保相同文本产生相同向量
        """
        start = time.time()
        
        # 模拟延迟（ms）
        base_latency = 1000 / self.speed
        jitter = np.random.normal(0, base_latency * 0.1)
        latency = max(10, base_latency + jitter)
        time.sleep(latency / 1000)
        
        # 生成确定性伪随机向量
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        vector = np.random.randn(self.dim).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist(), latency
    
    def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        """批量嵌入"""
        vectors = []
        total_latency = 0
        for text in texts:
            vec, lat = self.embed(text)
            vectors.append(vec)
            total_latency += lat
        return vectors, total_latency


# ============== 向量数据库（内存版） ==============

class VectorStore:
    """内存向量数据库"""
    
    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}
        self.texts: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add(self, doc_id: str, vector: List[float], text: str, metadata: Dict = None):
        self.vectors[doc_id] = np.array(vector)
        self.texts[doc_id] = text
        self.metadata[doc_id] = metadata or {}
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """余弦相似度搜索"""
        query = np.array(query_vector)
        query_norm = query / np.linalg.norm(query)
        
        scores = []
        for doc_id, vec in self.vectors.items():
            vec_norm = vec / np.linalg.norm(vec)
            similarity = np.dot(query_norm, vec_norm)
            scores.append((doc_id, float(similarity)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============== 评估指标 ==============

def compute_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """计算Recall@K"""
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)

def compute_mrr(retrieved: List[str], relevant: List[str]) -> float:
    """计算MRR"""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def compute_ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """计算NDCG@K"""
    relevance = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
    if not relevance or sum(relevance) == 0:
        return 0.0
    
    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    
    # IDCG
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    return dcg / idcg if idcg > 0 else 0.0


# ============== 测试数据 ==============

def load_financial_chunks(demo_path: str) -> List[Dict]:
    """加载chunking实验产生的文本块"""
    chunks = []
    demo_dir = Path(demo_path)
    
    # 加载分析文档
    for md_file in demo_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # 简单分块
        paragraphs = content.split('\n\n')
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append({
                    "id": f"{md_file.stem}_p{i}",
                    "text": para.strip(),
                    "source": md_file.name,
                    "type": "analysis"
                })
    
    # 加载JSON数据（简化版）
    json_dir = demo_dir / "mx_data" / "output"
    if json_dir.exists():
        for json_file in list(json_dir.glob("*_raw.json"))[:10]:  # 限制数量
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 提取关键字段作为文本
                text = json.dumps(data, ensure_ascii=False, indent=2)[:1000]
                chunks.append({
                    "id": json_file.stem,
                    "text": text,
                    "source": json_file.name,
                    "type": "financial_data"
                })
            except Exception as e:
                print(f"加载 {json_file} 失败: {e}")
    
    return chunks

def create_test_queries() -> List[Dict]:
    """创建测试查询集"""
    return [
        {
            "query": "沪电股份的ROE是多少",
            "relevant_docs": [],  # 将在运行时动态匹配
            "category": "财务指标"
        },
        {
            "query": "泰豪科技2025年净利润",
            "relevant_docs": [],
            "category": "年报数据"
        },
        {
            "query": "芯瑞达主力资金流向",
            "relevant_docs": [],
            "category": "资金流向"
        },
        {
            "query": "三安光电十大流通股东",
            "relevant_docs": [],
            "category": "股东信息"
        },
        {
            "query": "科瑞技术市盈率",
            "relevant_docs": [],
            "category": "估值指标"
        },
        {
            "query": "板块轮动涨停传导",
            "relevant_docs": [],
            "category": "分析文档"
        },
        {
            "query": "贝贝虾个股评分",
            "relevant_docs": [],
            "category": "分析文档"
        },
        {
            "query": "回测系统数据爬取",
            "relevant_docs": [],
            "category": "技术文档"
        },
    ]


# ============== 主实验流程 ==============

def run_embedding_benchmark():
    """运行Embedding模型对比实验"""
    print("=" * 80)
    print("Embedding模型对比实验")
    print("=" * 80)
    
    # 加载数据
    demo_path = os.path.expanduser("~/Desktop/agent_demo")
    chunks = load_financial_chunks(demo_path)
    print(f"\n📄 加载了 {len(chunks)} 个文本块")
    
    queries = create_test_queries()
    print(f"🎯 测试查询数: {len(queries)}")
    
    # 为每个查询匹配相关文档（基于关键词）
    for q in queries:
        keywords = q["query"].split()
        q["relevant_docs"] = [
            c["id"] for c in chunks
            if any(kw in c["text"] for kw in keywords) and len(c["text"]) > 20
        ][:5]  # 最多5个相关文档
        if not q["relevant_docs"]:
            # 如果没有精确匹配，放宽条件：匹配任意一个字
            q["relevant_docs"] = [
                c["id"] for c in chunks
                if any(kw[0] in c["text"] for kw in keywords if kw) and len(c["text"]) > 20
            ][:5]
    
    # 过滤掉没有相关文档的查询
    queries = [q for q in queries if q["relevant_docs"]]
    print(f"✅ 有效查询数: {len(queries)}")
    
    # 测试的模型
    models = ["text-embedding-3-small", "bge-large-zh", "m3e-base"]
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"🧪 测试模型: {model_name}")
        print("=" * 60)
        
        model = MockEmbeddingModel(model_name)
        store = VectorStore()
        
        # 1. 嵌入所有文档
        print("\n📊 嵌入文档中...")
        embed_times = []
        for chunk in chunks:
            vec, lat = model.embed(chunk["text"])
            store.add(chunk["id"], vec, chunk["text"], {
                "source": chunk["source"],
                "type": chunk["type"]
            })
            embed_times.append(lat)
        
        avg_embed_time = sum(embed_times) / len(embed_times)
        print(f"  平均嵌入耗时: {avg_embed_time:.1f}ms")
        print(f"  总文档数: {len(chunks)}")
        
        # 2. 执行检索测试
        print("\n🔍 执行检索测试...")
        recall_scores = defaultdict(list)
        mrr_scores = []
        ndcg_scores = defaultdict(list)
        query_times = []
        
        for q in queries:
            # 嵌入查询
            q_vec, q_lat = model.embed(q["query"])
            
            # 检索
            start = time.time()
            retrieved = store.search(q_vec, top_k=10)
            search_time = (time.time() - start) * 1000
            query_times.append(search_time + q_lat)
            
            retrieved_ids = [doc_id for doc_id, _ in retrieved]
            similarities = [score for _, score in retrieved]
            
            # 计算指标
            for k in [1, 3, 5, 10]:
                recall = compute_recall_at_k(retrieved_ids, q["relevant_docs"], k)
                recall_scores[k].append(recall)
                
                ndcg = compute_ndcg_at_k(retrieved_ids, q["relevant_docs"], k)
                ndcg_scores[k].append(ndcg)
            
            mrr = compute_mrr(retrieved_ids, q["relevant_docs"])
            mrr_scores.append(mrr)
            
            print(f"  [{q['category']}] {q['query'][:30]}...")
            print(f"    Recall@5: {recall_scores[5][-1]:.2f}, MRR: {mrr:.2f}")
        
        # 3. 汇总结果
        result = {
            "model": model_name,
            "dimension": model.dim,
            "num_docs": len(chunks),
            "recall_at_k": {k: sum(v)/len(v) for k, v in recall_scores.items()},
            "mrr": sum(mrr_scores) / len(mrr_scores),
            "ndcg_at_k": {k: sum(v)/len(v) for k, v in ndcg_scores.items()},
            "avg_embed_latency_ms": avg_embed_time,
            "avg_query_latency_ms": sum(query_times) / len(query_times),
        }
        results[model_name] = result
        
        print(f"\n📈 {model_name} 汇总:")
        print(f"  Recall@1: {result['recall_at_k'][1]:.3f}")
        print(f"  Recall@5: {result['recall_at_k'][5]:.3f}")
        print(f"  MRR: {result['mrr']:.3f}")
        print(f"  NDCG@5: {result['ndcg_at_k'][5]:.3f}")
        print(f"  平均查询延迟: {result['avg_query_latency_ms']:.1f}ms")
    
    # 4. 对比汇总
    print("\n" + "=" * 80)
    print("📊 模型对比汇总")
    print("=" * 80)
    print(f"{'模型':<25} {'维度':>6} {'R@1':>8} {'R@5':>8} {'MRR':>8} {'NDCG@5':>8} {'延迟(ms)':>10}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<25} {r['dimension']:>6} {r['recall_at_k'][1]:>8.3f} {r['recall_at_k'][5]:>8.3f} {r['mrr']:>8.3f} {r['ndcg_at_k'][5]:>8.3f} {r['avg_query_latency_ms']:>10.1f}")
    
    # 保存结果
    output_path = os.path.expanduser("~/Desktop/rag-experiments/embedding/results")
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/benchmark_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_path}/benchmark_results.json")
    
    return results


if __name__ == "__main__":
    run_embedding_benchmark()
