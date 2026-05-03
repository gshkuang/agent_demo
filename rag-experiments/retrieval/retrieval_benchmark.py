#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索策略对比实验
测试不同检索策略在金融文档RAG中的效果
"""

import json
import os
import time
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict


# ============== BM25实现（简化版） ==============

class SimpleBM25:
    """简化版BM25实现"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = {}
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.avgdl = 0
    
    def fit(self, docs: Dict[str, str]):
        """构建索引"""
        total_len = 0
        for doc_id, text in docs.items():
            words = self._tokenize(text)
            self.documents[doc_id] = words
            total_len += len(words)
            
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] += 1
        
        self.avgdl = total_len / len(docs) if docs else 0
        
        # 计算IDF
        N = len(docs)
        for word, df in self.doc_freqs.items():
            self.idf[word] = np.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        return re.findall(r'[\u4e00-\u9fff\w]+', text.lower())
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """BM25检索"""
        query_words = self._tokenize(query)
        scores = {}
        
        for doc_id, words in self.documents.items():
            score = 0
            for word in query_words:
                if word not in self.idf:
                    continue
                
                f = words.count(word)
                idf = self.idf[word]
                dl = len(words)
                
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator
            
            scores[doc_id] = score
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]


# ============== 向量检索（复用embedding实验的VectorStore） ==============

class SimpleVectorStore:
    """简化版向量存储"""
    
    def __init__(self):
        self.vectors = {}
        self.texts = {}
    
    def add(self, doc_id: str, vector: List[float], text: str):
        self.vectors[doc_id] = np.array(vector)
        self.texts[doc_id] = text
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        query = np.array(query_vector)
        query_norm = query / np.linalg.norm(query)
        
        scores = []
        for doc_id, vec in self.vectors.items():
            vec_norm = vec / np.linalg.norm(vec)
            similarity = np.dot(query_norm, vec_norm)
            scores.append((doc_id, float(similarity)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============== 检索策略 ==============

class RetrievalStrategies:
    """检索策略集合"""
    
    @staticmethod
    def vector_only(query_vec, vector_store, top_k=5):
        """纯向量检索"""
        return vector_store.search(query_vec, top_k)
    
    @staticmethod
    def bm25_only(query, bm25_index, top_k=5):
        """纯BM25检索"""
        return bm25_index.search(query, top_k)
    
    @staticmethod
    def hybrid_rrf(query, query_vec, vector_store, bm25_index, top_k=5, alpha=0.5, k_rrf=60):
        """
        混合检索 + RRF融合
        RRF: Reciprocal Rank Fusion
        score = sum(1 / (k + rank))
        """
        # 向量检索
        vec_results = vector_store.search(query_vec, top_k * 2)
        # BM25检索
        bm25_results = bm25_index.search(query, top_k * 2)
        
        # RRF融合
        rrf_scores = defaultdict(float)
        
        for rank, (doc_id, _) in enumerate(vec_results):
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank + 1)
        
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank + 1)
        
        # 排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    @staticmethod
    def hybrid_linear(query, query_vec, vector_store, bm25_index, top_k=5, alpha=0.5):
        """
        混合检索 + 线性融合
        score = alpha * vec_score + (1-alpha) * bm25_score
        """
        vec_results = vector_store.search(query_vec, top_k * 2)
        bm25_results = bm25_index.search(query, top_k * 2)
        
        # 归一化分数
        vec_dict = {doc_id: score for doc_id, score in vec_results}
        bm25_dict = {doc_id: score for doc_id, score in bm25_results}
        
        # 合并
        all_docs = set(vec_dict.keys()) | set(bm25_dict.keys())
        combined = {}
        
        for doc_id in all_docs:
            v_score = vec_dict.get(doc_id, 0)
            b_score = bm25_dict.get(doc_id, 0)
            
            # Min-Max归一化
            if vec_results:
                v_max = max(s for _, s in vec_results)
                v_min = min(s for _, s in vec_results)
                v_norm = (v_score - v_min) / (v_max - v_min) if v_max > v_min else 0
            else:
                v_norm = 0
            
            if bm25_results:
                b_max = max(s for _, s in bm25_results)
                b_min = min(s for _, s in bm25_results)
                b_norm = (b_score - b_min) / (b_max - b_min) if b_max > b_min else 0
            else:
                b_norm = 0
            
            combined[doc_id] = alpha * v_norm + (1 - alpha) * b_norm
        
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    @staticmethod
    def metadata_filter(query, query_vec, vector_store, bm25_index, metadata_store, top_k=5):
        """
        元数据预过滤 + 向量检索
        从查询中提取股票代码，先过滤再检索
        """
        # 提取股票代码（简化版）
        stock_codes = re.findall(r'\d{6}', query)
        
        if stock_codes:
            # 有过滤条件，先筛选文档
            filtered_docs = {
                doc_id: vec for doc_id, vec in vector_store.vectors.items()
                if doc_id.startswith(tuple(stock_codes)) or 
                   any(code in vector_store.texts.get(doc_id, '') for code in stock_codes)
            }
            
            if filtered_docs:
                scores = []
                for doc_id, vec in filtered_docs.items():
                    vec_norm = vec / np.linalg.norm(vec)
                    similarity = np.dot(query_vec, vec_norm)
                    scores.append((doc_id, float(similarity)))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                return scores[:top_k]
        
        # 无过滤条件，回退到纯向量检索
        return vector_store.search(query_vec, top_k)


# ============== 评估指标 ==============

def compute_metrics(retrieved: List[str], relevant: List[str]) -> Dict:
    """计算检索指标"""
    metrics = {}
    
    for k in [1, 3, 5, 10]:
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        # Recall@K
        metrics[f"recall@{k}"] = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
        
        # Precision@K
        metrics[f"precision@{k}"] = len(retrieved_set & relevant_set) / k if k > 0 else 0
    
    # MRR
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            metrics["mrr"] = 1.0 / (i + 1)
            break
    else:
        metrics["mrr"] = 0.0
    
    return metrics


# ============== 主实验流程 ==============

def run_retrieval_benchmark():
    """运行检索策略对比实验"""
    print("=" * 80)
    print("检索策略对比实验")
    print("=" * 80)
    
    # 加载数据
    demo_path = os.path.expanduser("~/Desktop/agent_demo")
    docs = {}
    
    # 加载Markdown文档（新路径）
    analysis_dir = Path(demo_path) / "data" / "analysis"
    for md_file in analysis_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            docs[md_file.stem] = f.read()
    
    # 加载JSON数据（新路径）
    json_dir = Path(demo_path) / "data" / "financial" / "mx_data" / "output"
    if json_dir.exists():
        for json_file in list(json_dir.glob("*_raw.json"))[:20]:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 提取文本内容用于embedding
                text = json.dumps(data, ensure_ascii=False, indent=2)
                docs[json_file.stem] = text[:3000]  # 限制长度
            except:
                pass
    
    print(f"\n📄 加载了 {len(docs)} 个文档")
    
    # 构建索引
    print("\n🔧 构建索引...")
    bm25 = SimpleBM25()
    bm25.fit(docs)
    
    # 使用真实BGE模型生成向量
    print("\n🔧 加载BGE模型生成真实向量...")
    from sentence_transformers import SentenceTransformer
    bge_model = SentenceTransformer("/tmp/models/BAAI/bge-large-zh-v1___5")
    
    vector_store = SimpleVectorStore()
    for doc_id, text in docs.items():
        vec = bge_model.encode(text, normalize_embeddings=True)
        vector_store.add(doc_id, vec.tolist(), text)
    
    print("  BM25索引构建完成")
    print("  向量索引构建完成（真实BGE语义向量）")
    
    # 测试查询
    queries = [
        {"query": "沪电股份 ROE", "relevant": ["5stocks-beibeixia-maomao-analysis_20260421"], "category": "财务指标"},
        {"query": "泰豪科技 净利润 2025", "relevant": ["mx_data_泰豪科技_营业收入_净利润_2023年_2024年_2025年_raw"], "category": "年报数据"},
        {"query": "芯瑞达 主力资金", "relevant": ["mx_data_芯瑞达_主力资金流向_近5日_raw"], "category": "资金流向"},
        {"query": "三安光电 流通股东", "relevant": ["mx_data_三安光电_十大流通股东_raw"], "category": "股东信息"},
        {"query": "科瑞技术 市盈率", "relevant": ["mx_data_科瑞技术_最新价_涨跌幅_市盈率_市净率_总市值_raw"], "category": "估值指标"},
        {"query": "板块轮动 涨停", "relevant": ["sector_rotation_20260418_review"], "category": "分析文档"},
        {"query": "贝贝虾 评分", "relevant": ["5stocks-beibeixia-maomao-analysis_20260421"], "category": "分析文档"},
        {"query": "回测系统 爬取", "relevant": ["四虾回测系统-数据爬取与回测_2026-05-02"], "category": "技术文档"},
        {"query": "苏大维格 资产负债率", "relevant": ["mx_data_三安光电,苏大维格,科瑞技术,华如科技_ROE_资产负债率_经营现金流_raw"], "category": "财务指标"},
        {"query": "华如科技 经营现金流", "relevant": ["mx_data_三安光电,苏大维格,科瑞技术,华如科技_ROE_资产负债率_经营现金流_raw"], "category": "财务指标"},
    ]
    
    # 测试策略
    strategies = {
        "纯向量检索": lambda q, qv: RetrievalStrategies.vector_only(qv, vector_store, 5),
        "纯BM25检索": lambda q, qv: RetrievalStrategies.bm25_only(q, bm25, 5),
        "混合检索(RRF)": lambda q, qv: RetrievalStrategies.hybrid_rrf(q, qv, vector_store, bm25, 5),
        "混合检索(线性, α=0.5)": lambda q, qv: RetrievalStrategies.hybrid_linear(q, qv, vector_store, bm25, 5, 0.5),
        "混合检索(线性, α=0.7)": lambda q, qv: RetrievalStrategies.hybrid_linear(q, qv, vector_store, bm25, 5, 0.7),
    }
    
    results = {}
    
    for strategy_name, strategy_fn in strategies.items():
        print(f"\n{'='*60}")
        print(f"🧪 测试策略: {strategy_name}")
        print("=" * 60)
        
        all_metrics = defaultdict(list)
        latencies = []
        
        for q in queries:
            # 使用BGE生成真实查询向量
            q_vec = bge_model.encode(q["query"], normalize_embeddings=True)
            
            # 执行检索
            start = time.time()
            retrieved = strategy_fn(q["query"], q_vec)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            retrieved_ids = [doc_id for doc_id, _ in retrieved]
            
            # 计算指标
            metrics = compute_metrics(retrieved_ids, q["relevant"])
            for k, v in metrics.items():
                all_metrics[k].append(v)
            
            print(f"  [{q['category']}] {q['query'][:30]}...")
            print(f"    Recall@5: {metrics['recall@5']:.2f}, MRR: {metrics['mrr']:.2f}, 延迟: {latency:.1f}ms")
        
        # 汇总
        result = {
            "strategy": strategy_name,
            "avg_recall@1": sum(all_metrics["recall@1"]) / len(all_metrics["recall@1"]),
            "avg_recall@5": sum(all_metrics["recall@5"]) / len(all_metrics["recall@5"]),
            "avg_precision@5": sum(all_metrics["precision@5"]) / len(all_metrics["precision@5"]),
            "avg_mrr": sum(all_metrics["mrr"]) / len(all_metrics["mrr"]),
            "avg_latency_ms": sum(latencies) / len(latencies),
        }
        results[strategy_name] = result
        
        print(f"\n📈 {strategy_name} 汇总:")
        print(f"  Recall@1: {result['avg_recall@1']:.3f}")
        print(f"  Recall@5: {result['avg_recall@5']:.3f}")
        print(f"  Precision@5: {result['avg_precision@5']:.3f}")
        print(f"  MRR: {result['avg_mrr']:.3f}")
        print(f"  平均延迟: {result['avg_latency_ms']:.1f}ms")
    
    # 对比汇总
    print("\n" + "=" * 80)
    print("📊 策略对比汇总")
    print("=" * 80)
    print(f"{'策略':<25} {'R@1':>8} {'R@5':>8} {'P@5':>8} {'MRR':>8} {'延迟(ms)':>10}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<25} {r['avg_recall@1']:>8.3f} {r['avg_recall@5']:>8.3f} {r['avg_precision@5']:>8.3f} {r['avg_mrr']:>8.3f} {r['avg_latency_ms']:>10.1f}")
    
    # 保存结果
    output_path = os.path.expanduser("~/Desktop/rag-experiments/retrieval/results")
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/benchmark_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_path}/benchmark_results.json")
    
    return results


if __name__ == "__main__":
    run_retrieval_benchmark()
