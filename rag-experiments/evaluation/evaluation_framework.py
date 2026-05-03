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


# ============== 模拟RAG系统组件 ==============

class MockChunker:
    """模拟分块器"""
    
    STRATEGIES = {
        "fixed": {"hit_rate": 0.85, "boundary_quality": 0.70},
        "recursive": {"hit_rate": 0.90, "boundary_quality": 1.00},
        "semantic": {"hit_rate": 0.88, "boundary_quality": 0.60},
        "structured": {"hit_rate": 0.92, "boundary_quality": 1.00},
    }
    
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.config = self.STRATEGIES.get(strategy, self.STRATEGIES["fixed"])
    
    def chunk(self, text: str) -> List[Dict]:
        """模拟分块"""
        # 根据策略返回不同质量的块
        words = text.split()
        chunk_size = 400 if self.strategy == "fixed" else 500
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunks.append({
                "id": f"chunk_{i}",
                "text": chunk_text,
                "strategy": self.strategy,
                "quality": self.config["boundary_quality"]
            })
        
        return chunks


class MockEmbedder:
    """模拟嵌入模型"""
    
    MODELS = {
        "text-embedding-3-small": {"dim": 1536, "quality": 0.90, "speed": 50},
        "text-embedding-3-large": {"dim": 3072, "quality": 0.95, "speed": 30},
        "bge-large-zh": {"dim": 1024, "quality": 0.88, "speed": 40},
        "m3e-base": {"dim": 768, "quality": 0.82, "speed": 60},
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = self.MODELS.get(model_name, self.MODELS["m3e-base"])
    
    def embed(self, text: str) -> Tuple[List[float], float]:
        """模拟嵌入"""
        latency = 1000 / self.config["speed"] + np.random.normal(0, 10)
        vec = np.random.randn(self.config["dim"])
        vec = vec / np.linalg.norm(vec)
        return vec.tolist(), max(10, latency)


class MockRetriever:
    """模拟检索器"""
    
    STRATEGIES = {
        "vector": {"recall": 0.75, "precision": 0.65},
        "bm25": {"recall": 0.70, "precision": 0.70},
        "hybrid_rrf": {"recall": 0.85, "precision": 0.75},
        "hybrid_linear": {"recall": 0.82, "precision": 0.72},
    }
    
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.config = self.STRATEGIES.get(strategy, self.STRATEGIES["vector"])
    
    def retrieve(self, query: str, chunks: List[Dict], top_k: int = 5) -> Tuple[List[Dict], float]:
        """模拟检索"""
        latency = np.random.normal(100, 20)
        
        # 根据策略质量返回结果
        recall = self.config["recall"]
        num_relevant = int(len(chunks) * recall * 0.3)
        
        # 模拟检索结果
        results = []
        for i, chunk in enumerate(chunks[:top_k]):
            is_relevant = i < num_relevant
            results.append({
                "chunk": chunk,
                "score": 0.8 - i * 0.1,
                "is_relevant": is_relevant
            })
        
        return results, max(50, latency)


class MockGenerator:
    """模拟生成器"""
    
    def generate(self, query: str, retrieved_chunks: List[Dict]) -> Tuple[str, float]:
        """模拟生成回答"""
        # 基于检索质量模拟生成质量
        avg_chunk_quality = np.mean([r["chunk"].get("quality", 0.8) for r in retrieved_chunks]) if retrieved_chunks else 0.5
        has_relevant = any(r.get("is_relevant", False) for r in retrieved_chunks)
        
        # 生成延迟
        latency = np.random.normal(2000, 500)
        
        # 模拟回答质量
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
    
    # 加载文档
    demo_path = os.path.expanduser("~/Desktop/agent_demo")
    documents = {}
    for md_file in Path(demo_path).glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            documents[md_file.stem] = f.read()
    
    print(f"📚 加载了 {len(documents)} 个文档")
    
    # 定义测试配置矩阵
    configs = [
        RAGConfig("fixed", "m3e-base", "vector"),
        RAGConfig("recursive", "m3e-base", "vector"),
        RAGConfig("structured", "m3e-base", "vector"),
        RAGConfig("recursive", "text-embedding-3-small", "vector"),
        RAGConfig("recursive", "m3e-base", "bm25"),
        RAGConfig("recursive", "m3e-base", "hybrid_rrf"),
        RAGConfig("recursive", "m3e-base", "hybrid_linear"),
        RAGConfig("structured", "text-embedding-3-small", "hybrid_rrf"),
    ]
    
    results = []
    
    for config in configs:
        config_name = f"{config.chunking_strategy}+{config.embedding_model}+{config.retrieval_strategy}"
        print(f"\n{'='*70}")
        print(f"🧪 测试配置: {config_name}")
        print("=" * 70)
        
        # 初始化组件
        chunker = MockChunker(config.chunking_strategy)
        embedder = MockEmbedder(config.embedding_model)
        retriever = MockRetriever(config.retrieval_strategy)
        generator = MockGenerator()
        
        # 预处理文档
        all_chunks = []
        for doc_id, text in documents.items():
            chunks = chunker.chunk(text)
            for chunk in chunks:
                chunk["doc_id"] = doc_id
            all_chunks.extend(chunks)
        
        print(f"  总分块数: {len(all_chunks)}")
        
        # 评估每个问答对
        metrics_list = []
        
        for qa in qa_pairs:
            # 1. 检索
            retrieved, ret_latency = retriever.retrieve(qa.question, all_chunks, config.top_k)
            
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
