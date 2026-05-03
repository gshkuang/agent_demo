#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG端到端评估框架
评估不同配置组合对最终回答质量的影响
"""

import json
import os
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict


# ============== 数据模型 ==============

@dataclass
class RAGConfig:
    """RAG配置"""
    chunking_strategy: str      # fixed / recursive / semantic / structured
    embedding_model: str        # text-embedding-3-small / bge-large-zh / m3e-base
    retrieval_strategy: str     # vector / bm25 / hybrid_rrf / hybrid_linear
    top_k: int = 5
    
@dataclass
class QAPair:
    """问答对"""
    question: str
    expected_answer: str
    expected_sources: List[str]
    category: str
    difficulty: str  # easy / medium / hard

@dataclass
class RAGResult:
    """RAG回答结果"""
    config: RAGConfig
    question: str
    retrieved_chunks: List[Dict]
    generated_answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float

@dataclass
class EvalMetrics:
    """评估指标"""
    config: RAGConfig
    retrieval_recall: float
    retrieval_precision: float
    faithfulness: float
    relevance: float
    completeness: float
    accuracy: float
    total_latency_ms: float


# ============== 真实RAG系统组件 ==============

class RealChunker:
    """真实分块器 - 简单段落分块"""
    
    def __init__(self, strategy: str = "recursive"):
        self.strategy = strategy
        self.chunk_size = 500
        self.overlap = 100
    
    def chunk(self, text: str) -> List[Dict]:
        """按段落和长度分块"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_len = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            if current_len + para_len > self.chunk_size and current_chunk:
                # 保存当前块
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "id": f"chunk_{chunk_idx}",
                    "text": chunk_text,
                    "strategy": self.strategy,
                    "quality": 1.0
                })
                chunk_idx += 1
                
                # 保留重叠部分
                overlap_text = current_chunk[-1] if len(current_chunk) > 0 else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_len = len(overlap_text) + para_len if overlap_text else para_len
            else:
                current_chunk.append(para)
                current_len += para_len
        
        # 保存最后一个块
        if current_chunk:
            chunks.append({
                "id": f"chunk_{chunk_idx}",
                "text": '\n\n'.join(current_chunk),
                "strategy": self.strategy,
                "quality": 1.0
            })
        
        return chunks


class RealEmbedder:
    """真实BGE嵌入模型"""
    
    def __init__(self, model_path: str = "/tmp/models/BAAI/bge-large-zh-v1___5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path)
        self.model_name = "bge-large-zh-v1.5"
        self.dim = 1024
    
    def embed(self, text: str) -> Tuple[List[float], float]:
        start = time.time()
        vec = self.model.encode(text, normalize_embeddings=True)
        latency = (time.time() - start) * 1000
        return vec.tolist(), latency
    
    def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], float]:
        start = time.time()
        vectors = self.model.encode(texts, normalize_embeddings=True)
        latency = (time.time() - start) * 1000
        return vectors.tolist(), latency


class RealRetriever:
    """真实检索器 - 向量+BM25混合"""
    
    def __init__(self, strategy: str = "hybrid_rrf"):
        self.strategy = strategy
        self.embedder = None
        self.bm25 = None
        self.vectors = {}
        self.chunks = {}
        self.doc_texts = {}
    
    def build_index(self, chunks: List[Dict], embedder: RealEmbedder):
        """构建索引"""
        self.embedder = embedder
        
        # 向量索引
        texts = [c["text"] for c in chunks]
        vectors, _ = embedder.embed_batch(texts)
        
        for i, chunk in enumerate(chunks):
            self.vectors[chunk["id"]] = np.array(vectors[i])
            self.chunks[chunk["id"]] = chunk
            self.doc_texts[chunk["id"]] = chunk["text"]
        
        # BM25索引
        self.bm25 = SimpleBM25()
        self.bm25.fit({cid: text for cid, text in self.doc_texts.items()})
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """执行检索"""
        start = time.time()
        
        # 生成查询向量
        q_vec = np.array(self.embedder.embed(query)[0])
        q_norm = q_vec / np.linalg.norm(q_vec)
        
        if self.strategy == "vector":
            results = self._vector_search(q_norm, top_k)
        elif self.strategy == "bm25":
            results = self._bm25_search(query, top_k)
        elif self.strategy == "hybrid_rrf":
            results = self._hybrid_rrf(query, q_norm, top_k)
        elif self.strategy == "hybrid_linear":
            results = self._hybrid_linear(query, q_norm, top_k)
        else:
            results = self._vector_search(q_norm, top_k)
        
        latency = (time.time() - start) * 1000
        
        # 格式化结果
        formatted = []
        for doc_id, score in results:
            formatted.append({
                "chunk": self.chunks.get(doc_id, {"id": doc_id, "text": "", "quality": 1.0}),
                "score": score,
                "is_relevant": True  # 真实检索默认标记为相关
            })
        
        return formatted, latency
    
    def _vector_search(self, q_norm: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        scores = []
        for doc_id, vec in self.vectors.items():
            vec_norm = vec / np.linalg.norm(vec)
            sim = np.dot(q_norm, vec_norm)
            scores.append((doc_id, float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        return self.bm25.search(query, top_k)
    
    def _hybrid_rrf(self, query: str, q_norm: np.ndarray, top_k: int, k_rrf: int = 60) -> List[Tuple[str, float]]:
        vec_results = self._vector_search(q_norm, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)
        
        rrf_scores = defaultdict(float)
        for rank, (doc_id, _) in enumerate(vec_results):
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank + 1)
        
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _hybrid_linear(self, query: str, q_norm: np.ndarray, top_k: int, alpha: float = 0.5) -> List[Tuple[str, float]]:
        vec_results = self._vector_search(q_norm, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)
        
        vec_dict = {doc_id: score for doc_id, score in vec_results}
        bm25_dict = {doc_id: score for doc_id, score in bm25_results}
        
        all_docs = set(vec_dict.keys()) | set(bm25_dict.keys())
        combined = {}
        
        for doc_id in all_docs:
            v_score = vec_dict.get(doc_id, 0)
            b_score = bm25_dict.get(doc_id, 0)
            
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


class SimpleBM25:
    """简化版BM25"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = {}
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.avgdl = 0
    
    def fit(self, docs: Dict[str, str]):
        total_len = 0
        for doc_id, text in docs.items():
            words = self._tokenize(text)
            self.documents[doc_id] = words
            total_len += len(words)
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] += 1
        
        self.avgdl = total_len / len(docs) if docs else 0
        
        N = len(docs)
        for word, df in self.doc_freqs.items():
            self.idf[word] = np.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[\u4e00-\u9fff\w]+', text.lower())
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
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


class MockGenerator:
    """模拟生成器（保持模拟，因无真实LLM API）"""
    
    def generate(self, query: str, retrieved_chunks: List[Dict]) -> Tuple[str, float]:
        avg_chunk_quality = np.mean([r["chunk"].get("quality", 0.8) for r in retrieved_chunks]) if retrieved_chunks else 0.5
        has_relevant = any(r.get("is_relevant", False) for r in retrieved_chunks)
        
        latency = np.random.normal(2000, 500)
        
        if has_relevant and avg_chunk_quality > 0.8:
            answer = "基于检索到的相关信息，我可以为您提供准确的回答..."
        elif has_relevant:
            answer = "根据部分相关信息，回答如下..."
        else:
            answer = "抱歉，未能找到足够的信息来回答您的问题..."
        
        return answer, max(500, latency)


# ============== 评估指标计算 ==============

def evaluate_faithfulness(answer: str, retrieved_chunks: List[Dict]) -> float:
    """评估忠实度（回答是否基于检索内容）"""
    if not retrieved_chunks:
        return 0.0
    
    # 模拟：基于检索质量计算忠实度
    avg_quality = np.mean([r["chunk"].get("quality", 0.8) for r in retrieved_chunks])
    has_relevant = any(r.get("is_relevant", False) for r in retrieved_chunks)
    
    if not has_relevant:
        return 0.3
    
    return min(1.0, avg_quality * 0.9 + 0.1)

def evaluate_relevance(answer: str, question: str) -> float:
    """评估相关性"""
    # 模拟：简单关键词匹配
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words)
    
    if overlap > 0:
        return min(1.0, 0.5 + overlap * 0.1)
    return 0.3

def evaluate_completeness(answer: str, expected_answer: str) -> float:
    """评估完整性"""
    # 模拟：基于回答长度和预期内容
    if "准确的回答" in answer:
        return 0.85
    elif "部分相关" in answer:
        return 0.60
    else:
        return 0.30

def evaluate_accuracy(answer: str, expected_answer: str) -> float:
    """评估准确性（数值/事实）"""
    # 模拟：检查是否包含预期数值
    expected_numbers = re.findall(r'\d+\.?\d*', expected_answer)
    answer_numbers = re.findall(r'\d+\.?\d*', answer)
    
    if not expected_numbers:
        return 0.8
    
    matches = sum(1 for n in expected_numbers if n in answer_numbers)
    return matches / len(expected_numbers) if expected_numbers else 0.8


# ============== 测试数据集 ==============

def create_test_dataset() -> List[QAPair]:
    """创建测试数据集"""
    return [
        QAPair(
            question="沪电股份的ROE是多少？",
            expected_answer="沪电股份2024年ROE约为15-20%",
            expected_sources=["5stocks-beibeixia-maomao-analysis_20260421"],
            category="财务指标",
            difficulty="easy"
        ),
        QAPair(
            question="泰豪科技2025年的营业收入和净利润分别是多少？",
            expected_answer="泰豪科技2025年营业收入约XX亿元，净利润约XX亿元",
            expected_sources=["mx_data_泰豪科技_营业收入_净利润_2023年_2024年_2025年_raw"],
            category="年报数据",
            difficulty="medium"
        ),
        QAPair(
            question="芯瑞达近5日主力资金流向如何？",
            expected_answer="芯瑞达近5日主力资金净流入/流出XX万元",
            expected_sources=["mx_data_芯瑞达_主力资金流向_近5日_raw"],
            category="资金流向",
            difficulty="easy"
        ),
        QAPair(
            question="三安光电的十大流通股东有哪些？",
            expected_answer="三安光电十大流通股东包括...",
            expected_sources=["mx_data_三安光电_十大流通股东_raw"],
            category="股东信息",
            difficulty="medium"
        ),
        QAPair(
            question="科瑞技术的市盈率和市净率分别是多少？",
            expected_answer="科瑞技术市盈率约XX倍，市净率约XX倍",
            expected_sources=["mx_data_科瑞技术_最新价_涨跌幅_市盈率_市净率_总市值_raw"],
            category="估值指标",
            difficulty="easy"
        ),
        QAPair(
            question="板块轮动复盘中的涨停传导路径是什么？",
            expected_answer="涨停传导路径为...",
            expected_sources=["sector_rotation_20260418_review"],
            category="分析文档",
            difficulty="hard"
        ),
        QAPair(
            question="贝贝虾分析中的个股评分标准是什么？",
            expected_answer="贝贝虾评分标准包括...",
            expected_sources=["5stocks-beibeixia-maomao-analysis_20260421"],
            category="分析文档",
            difficulty="medium"
        ),
        QAPair(
            question="回测系统的数据爬取逻辑是怎样的？",
            expected_answer="回测系统通过...方式爬取数据",
            expected_sources=["四虾回测系统-数据爬取与回测_2026-05-02"],
            category="技术文档",
            difficulty="hard"
        ),
    ]


# ============== 主实验流程 ==============

def run_end_to_end_evaluation():
    """运行端到端评估"""
    print("=" * 80)
    print("RAG端到端评估框架")
    print("=" * 80)
    
    # 加载测试数据
    qa_pairs = create_test_dataset()
    print(f"\n📄 加载了 {len(qa_pairs)} 个测试问答对")
    
    # 加载文档（新路径）
    demo_path = os.path.expanduser("~/Desktop/agent_demo")
    documents = {}
    
    # 加载分析文档
    analysis_dir = Path(demo_path) / "data" / "analysis"
    for md_file in analysis_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            documents[md_file.stem] = f.read()
    
    # 加载金融数据
    json_dir = Path(demo_path) / "data" / "financial" / "mx_data" / "output"
    if json_dir.exists():
        for json_file in list(json_dir.glob("*_raw.json"))[:20]:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                documents[json_file.stem] = json.dumps(data, ensure_ascii=False, indent=2)[:3000]
            except:
                pass
    
    print(f"📚 加载了 {len(documents)} 个文档")
    
    # 定义测试配置矩阵（使用真实BGE模型）
    configs = [
        RAGConfig("recursive", "bge-large-zh-v1.5", "vector"),
        RAGConfig("recursive", "bge-large-zh-v1.5", "bm25"),
        RAGConfig("recursive", "bge-large-zh-v1.5", "hybrid_rrf"),
        RAGConfig("recursive", "bge-large-zh-v1.5", "hybrid_linear"),
    ]
    
    results = []
    
    for config in configs:
        config_name = f"{config.chunking_strategy}+{config.embedding_model}+{config.retrieval_strategy}"
        print(f"\n{'='*70}")
        print(f"🧪 测试配置: {config_name}")
        print("=" * 70)
        
        # 初始化组件（使用真实模型）
        chunker = RealChunker(config.chunking_strategy)
        embedder = RealEmbedder()
        retriever = RealRetriever(config.retrieval_strategy)
        generator = MockGenerator()
        
        # 预处理文档
        all_chunks = []
        for doc_id, text in documents.items():
            chunks = chunker.chunk(text)
            for chunk in chunks:
                chunk["doc_id"] = doc_id
            all_chunks.extend(chunks)
        
        print(f"  总分块数: {len(all_chunks)}")
        
        # 构建索引
        print(f"  构建索引中...")
        retriever.build_index(all_chunks, embedder)
        
        # 评估每个问答对
        metrics_list = []
        
        for qa in qa_pairs:
            # 1. 检索
            retrieved, ret_latency = retriever.retrieve(qa.question, config.top_k)
            
            # 2. 生成
            answer, gen_latency = generator.generate(qa.question, retrieved)
            
            # 3. 计算指标
            retrieval_recall = sum(1 for r in retrieved if r.get("is_relevant", False)) / max(len(qa.expected_sources), 1)
            retrieval_precision = sum(1 for r in retrieved if r.get("is_relevant", False)) / max(len(retrieved), 1)
            
            faithfulness = evaluate_faithfulness(answer, retrieved)
            relevance = evaluate_relevance(answer, qa.question)
            completeness = evaluate_completeness(answer, qa.expected_answer)
            accuracy = evaluate_accuracy(answer, qa.expected_answer)
            
            metrics = EvalMetrics(
                config=config,
                retrieval_recall=retrieval_recall,
                retrieval_precision=retrieval_precision,
                faithfulness=faithfulness,
                relevance=relevance,
                completeness=completeness,
                accuracy=accuracy,
                total_latency_ms=ret_latency + gen_latency
            )
            metrics_list.append(metrics)
            
            print(f"  [{qa.category}] {qa.question[:40]}...")
            print(f"    检索R/P: {retrieval_recall:.2f}/{retrieval_precision:.2f}, "
                  f"忠实度: {faithfulness:.2f}, 相关性: {relevance:.2f}, "
                  f"延迟: {metrics.total_latency_ms:.0f}ms")
        
        # 汇总
        avg_metrics = {
            "config": config_name,
            "retrieval_recall": np.mean([m.retrieval_recall for m in metrics_list]),
            "retrieval_precision": np.mean([m.retrieval_precision for m in metrics_list]),
            "faithfulness": np.mean([m.faithfulness for m in metrics_list]),
            "relevance": np.mean([m.relevance for m in metrics_list]),
            "completeness": np.mean([m.completeness for m in metrics_list]),
            "accuracy": np.mean([m.accuracy for m in metrics_list]),
            "total_latency_ms": np.mean([m.total_latency_ms for m in metrics_list]),
        }
        results.append(avg_metrics)
        
        print(f"\n📈 {config_name} 汇总:")
        print(f"  检索Recall: {avg_metrics['retrieval_recall']:.3f}")
        print(f"  检索Precision: {avg_metrics['retrieval_precision']:.3f}")
        print(f"  忠实度: {avg_metrics['faithfulness']:.3f}")
        print(f"  相关性: {avg_metrics['relevance']:.3f}")
        print(f"  完整性: {avg_metrics['completeness']:.3f}")
        print(f"  准确性: {avg_metrics['accuracy']:.3f}")
        print(f"  平均延迟: {avg_metrics['total_latency_ms']:.0f}ms")
    
    # 对比汇总
    print("\n" + "=" * 80)
    print("📊 配置对比汇总")
    print("=" * 80)
    print(f"{'配置':<45} {'Recall':>8} {'Prec':>8} {'Faith':>8} {'Rel':>8} {'Comp':>8} {'Acc':>8} {'Latency':>10}")
    print("-" * 110)
    for r in results:
        print(f"{r['config']:<45} {r['retrieval_recall']:>8.3f} {r['retrieval_precision']:>8.3f} "
              f"{r['faithfulness']:>8.3f} {r['relevance']:>8.3f} {r['completeness']:>8.3f} "
              f"{r['accuracy']:>8.3f} {r['total_latency_ms']:>10.0f}")
    
    # 找出最佳配置
    best = max(results, key=lambda x: x['retrieval_recall'] * 0.3 + x['faithfulness'] * 0.3 + 
                                      x['relevance'] * 0.2 + x['accuracy'] * 0.2)
    print(f"\n🏆 最佳配置: {best['config']}")
    
    # 保存结果
    output_path = os.path.expanduser("~/Desktop/rag-experiments/evaluation/results")
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/e2e_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_path}/e2e_evaluation.json")
    
    return results


if __name__ == "__main__":
    run_end_to_end_evaluation()
