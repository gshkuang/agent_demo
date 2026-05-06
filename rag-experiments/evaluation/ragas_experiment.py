#!/usr/bin/env python3
"""
RAGAS对比实验 - 不同RAG配置的评测
对比: 分块策略 × 嵌入模型 × 检索策略
"""
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

from ragas import evaluate
# RAGAS 0.4.x 指标导入（兼容未来版本，优先使用新路径避免弃用警告）
try:
    from ragas.metrics.collections import (
        faithfulness, answer_relevancy, context_precision, context_recall
    )
except ImportError:
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall
    )
from ragas.dataset_schema import SingleTurnSample

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_community.retrievers import BM25Retriever
# EnsembleRetriever - 尝试多种导入路径
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        # 如果都没有，使用简单的加权混合实现
        class EnsembleRetriever:
            """简单的EnsembleRetriever替代实现"""
            def __init__(self, retrievers, weights):
                self.retrievers = retrievers
                self.weights = weights
            
            def invoke(self, query):
                from collections import OrderedDict
                all_docs = OrderedDict()
                for retriever, weight in zip(self.retrievers, self.weights):
                    docs = retriever.invoke(query)
                    for doc in docs:
                        key = doc.page_content
                        if key not in all_docs:
                            all_docs[key] = (doc, weight)
                        else:
                            # 累加权重
                            old_doc, old_weight = all_docs[key]
                            all_docs[key] = (old_doc, old_weight + weight)
                
                # 按权重排序
                sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in sorted_docs[:5]]
from langchain_core.documents import Document


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    chunk_strategy: str  # fixed, recursive, markdown
    embedding_model: str  # openai, bge
    retrieval_strategy: str  # vector, bm25, hybrid


class RagasExperiment:
    """RAGAS对比实验"""
    
    def __init__(self, llm_model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.results = []
    
    def load_documents(self, path: str) -> List[Document]:
        """加载文档"""
        docs = []
        p = Path(path)
        
        # Markdown文件
        for f in p.glob("*.md"):
            docs.append(Document(
                page_content=f.read_text(encoding="utf-8"),
                metadata={"source": f.name, "type": "md"}
            ))
        
        # JSON数据文件
        json_dir = p / "mx_data" / "output"
        if json_dir.exists():
            for f in list(json_dir.glob("*_raw.json"))[:20]:
                docs.append(Document(
                    page_content=json.dumps(json.load(f.open()), ensure_ascii=False),
                    metadata={"source": f.name, "type": "json"}
                ))
        
        return docs
    
    def chunk_documents(self, docs: List[Document], strategy: str) -> List[Document]:
        """文档分块"""
        if strategy == "fixed":
            splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=400,
                chunk_overlap=50
            )
        elif strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
        elif strategy == "markdown":
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
            )
            # MarkdownHeaderTextSplitter需要特殊处理
            all_chunks = []
            for doc in docs:
                if doc.metadata.get("type") == "md":
                    chunks = splitter.split_text(doc.page_content)
                    all_chunks.extend([
                        Document(page_content=c, metadata=doc.metadata)
                        for c in chunks
                    ])
                else:
                    all_chunks.append(doc)
            return all_chunks
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        
        return splitter.split_documents(docs)
    
    def get_embedder(self, model_name: str):
        """获取嵌入模型"""
        if model_name == "bge":
            return HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
        elif model_name == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            return HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    
    def build_retriever(self, chunks: List[Document], embedder, strategy: str):
        """构建检索器"""
        if strategy == "bm25":
            return BM25Retriever.from_documents(chunks, k=5)
        
        # 向量检索
        vector_store = Chroma.from_documents(chunks, embedder)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        if strategy == "vector":
            return vector_retriever
        elif strategy == "hybrid":
            bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
            return EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
        
        return vector_retriever
    
    def evaluate_config(self, config: ExperimentConfig, qa_pairs: List[Dict], docs: List[Document]) -> Dict:
        """评测单个配置"""
        print(f"\n{'='*50}")
        print(f"配置: {config.name}")
        print(f"  分块: {config.chunk_strategy}")
        print(f"  嵌入: {config.embedding_model}")
        print(f"  检索: {config.retrieval_strategy}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # 1. 分块
        chunk_start = time.time()
        chunks = self.chunk_documents(docs, config.chunk_strategy)
        chunk_time = (time.time() - chunk_start) * 1000
        print(f"分块: {len(chunks)} chunks, {chunk_time:.0f}ms")
        
        # 2. 嵌入
        embedder = self.get_embedder(config.embedding_model)
        
        # 3. 构建检索器
        retriever = self.build_retriever(chunks, embedder, config.retrieval_strategy)
        
        # 4. 评测每个QA对
        results = []
        for qa in qa_pairs:
            # 检索
            ret_start = time.time()
            retrieved_docs = retriever.invoke(qa["question"])
            ret_time = (time.time() - ret_start) * 1000
            
            contexts = [d.page_content for d in retrieved_docs[:5]]
            
            # 生成回答
            context_text = "\n\n".join(contexts[:3])
            prompt = f"基于以下上下文回答问题:\n{context_text}\n\n问题: {qa['question']}\n回答:"
            
            gen_start = time.time()
            answer = self.llm.invoke(prompt).content
            gen_time = (time.time() - gen_start) * 1000
            
            # RAGAS评估
            sample = SingleTurnSample(
                user_input=qa["question"],
                response=answer,
                reference=qa["ground_truth"],
                retrieved_contexts=contexts
            )
            
            metrics = {}
            for metric_name, metric_func in [
                ("faithfulness", faithfulness),
                ("answer_relevancy", answer_relevancy),
                ("context_precision", context_precision),
                ("context_recall", context_recall)
            ]:
                try:
                    metrics[metric_name] = metric_func.single_turn_score(sample)
                except:
                    metrics[metric_name] = 0.0
            
            results.append({
                "question": qa["question"],
                "answer": answer[:200],
                **metrics,
                "retrieval_ms": ret_time,
                "generation_ms": gen_time,
                "contexts_count": len(contexts)
            })
            
            print(f"  {qa['question'][:40]}... "
                  f"F:{metrics['faithfulness']:.2f} "
                  f"R:{metrics['answer_relevancy']:.2f} "
                  f"P:{metrics['context_precision']:.2f} "
                  f"Rec:{metrics['context_recall']:.2f}")
        
        total_time = (time.time() - start_time) * 1000
        
        # 汇总
        avg_metrics = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            scores = [r[metric] for r in results]
            avg_metrics[metric] = sum(scores) / len(scores) if scores else 0
        
        overall = sum(avg_metrics.values()) / len(avg_metrics) if avg_metrics else 0
        
        return {
            "config": {
                "name": config.name,
                "chunk_strategy": config.chunk_strategy,
                "embedding_model": config.embedding_model,
                "retrieval_strategy": config.retrieval_strategy
            },
            "overall_score": overall,
            "metrics": avg_metrics,
            "avg_retrieval_ms": sum(r["retrieval_ms"] for r in results) / len(results),
            "avg_generation_ms": sum(r["generation_ms"] for r in results) / len(results),
            "total_time_ms": total_time,
            "chunks_count": len(chunks),
            "details": results
        }
    
    def run_experiments(self, configs: List[ExperimentConfig], qa_pairs: List[Dict], docs: List[Document]):
        """运行所有实验"""
        print("=" * 60)
        print("RAGAS 对比实验")
        print("=" * 60)
        print(f"配置数: {len(configs)}")
        print(f"QA对数: {len(qa_pairs)}")
        print(f"文档数: {len(docs)}")
        
        all_results = []
        for config in configs:
            try:
                result = self.evaluate_config(config, qa_pairs, docs)
                all_results.append(result)
            except Exception as e:
                print(f"\n⚠️ 配置 {config.name} 失败: {e}")
                all_results.append({
                    "config": config.__dict__,
                    "error": str(e)
                })
        
        return all_results
    
    def generate_report(self, results: List[Dict]):
        """生成实验报告"""
        print("\n" + "=" * 70)
        print("📊 RAGAS 对比实验报告")
        print("=" * 70)
        
        # 排序
        valid_results = [r for r in results if "overall_score" in r]
        valid_results.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # 表格
        print("\n【配置排名】")
        print(f"{'排名':<4} {'配置':<30} {'综合分':<8} {'忠实度':<8} {'相关性':<8} {'精确率':<8} {'召回率':<8} {'延迟(ms)':<10}")
        print("-" * 90)
        
        for i, r in enumerate(valid_results, 1):
            c = r["config"]
            m = r["metrics"]
            name = f"{c['chunk_strategy']}+{c['embedding_model']}+{c['retrieval_strategy']}"
            print(f"{i:<4} {name:<30} {r['overall_score']:.3f}    "
                  f"{m.get('faithfulness', 0):.3f}    "
                  f"{m.get('answer_relevancy', 0):.3f}    "
                  f"{m.get('context_precision', 0):.3f}    "
                  f"{m.get('context_recall', 0):.3f}    "
                  f"{r.get('avg_retrieval_ms', 0)+r.get('avg_generation_ms', 0):.0f}")
        
        # 最佳配置
        if valid_results:
            best = valid_results[0]
            print(f"\n🏆 最佳配置: {best['config']['name']}")
            print(f"   综合评分: {best['overall_score']:.3f}")
            print(f"   分块策略: {best['config']['chunk_strategy']}")
            print(f"   嵌入模型: {best['config']['embedding_model']}")
            print(f"   检索策略: {best['config']['retrieval_strategy']}")
        
        # 维度分析
        print("\n【维度分析】")
        
        # 分块策略对比
        chunk_scores = {}
        for r in valid_results:
            strategy = r["config"]["chunk_strategy"]
            if strategy not in chunk_scores:
                chunk_scores[strategy] = []
            chunk_scores[strategy].append(r["overall_score"])
        
        print("\n分块策略:")
        for strategy, scores in chunk_scores.items():
            avg = sum(scores) / len(scores)
            print(f"  {strategy:12s}: {avg:.3f} (n={len(scores)})")
        
        # 嵌入模型对比
        embed_scores = {}
        for r in valid_results:
            model = r["config"]["embedding_model"]
            if model not in embed_scores:
                embed_scores[model] = []
            embed_scores[model].append(r["overall_score"])
        
        print("\n嵌入模型:")
        for model, scores in embed_scores.items():
            avg = sum(scores) / len(scores)
            print(f"  {model:12s}: {avg:.3f} (n={len(scores)})")
        
        # 检索策略对比
        ret_scores = {}
        for r in valid_results:
            strategy = r["config"]["retrieval_strategy"]
            if strategy not in ret_scores:
                ret_scores[strategy] = []
            ret_scores[strategy].append(r["overall_score"])
        
        print("\n检索策略:")
        for strategy, scores in ret_scores.items():
            avg = sum(scores) / len(scores)
            print(f"  {strategy:12s}: {avg:.3f} (n={len(scores)})")
        
        return valid_results


def create_test_data() -> List[Dict]:
    """创建测试QA对"""
    return [
        {
            "question": "沪电股份2024年的ROE是多少？",
            "ground_truth": "沪电股份2024年ROE为18.5%，在PCB行业中处于领先水平。"
        },
        {
            "question": "泰豪科技2025年净利润预计多少？",
            "ground_truth": "泰豪科技2025年预计净利润XX亿元，同比增长XX%。"
        },
        {
            "question": "芯瑞达最近主力资金流向如何？",
            "ground_truth": "芯瑞达近期主力资金呈净流入状态，近5日净流入约XX万元。"
        },
        {
            "question": "半导体板块最近的涨停传导路径是什么？",
            "ground_truth": "半导体板块涨停传导路径为：设备→材料→设计→封测。"
        },
        {
            "question": "贝贝虾评分系统包含哪些维度？",
            "ground_truth": "贝贝虾评分系统包含市场环境、板块情绪、个股交易结构三个维度。"
        }
    ]


def main():
    """主函数"""
    # 实验配置矩阵
    configs = [
        ExperimentConfig("fixed+bge+vector", "fixed", "bge", "vector"),
        ExperimentConfig("recursive+bge+vector", "recursive", "bge", "vector"),
        ExperimentConfig("recursive+openai+vector", "recursive", "openai", "vector"),
        ExperimentConfig("recursive+bge+bm25", "recursive", "bge", "bm25"),
        ExperimentConfig("recursive+bge+hybrid", "recursive", "bge", "hybrid"),
        ExperimentConfig("markdown+bge+vector", "markdown", "bge", "vector"),
    ]
    
    # 初始化实验
    experiment = RagasExperiment(llm_model="gpt-4o")
    
    # 加载文档
    docs_path = os.path.expanduser("~/Desktop/agent_demo")
    docs = experiment.load_documents(docs_path)
    print(f"加载 {len(docs)} 个文档")
    
    # 加载QA对
    qa_pairs = create_test_data()
    
    # 运行实验
    results = experiment.run_experiments(configs, qa_pairs, docs)
    
    # 生成报告
    ranked_results = experiment.generate_report(results)
    
    # 保存结果
    out_dir = Path(os.path.expanduser("~/Desktop/agent_demo/rag-experiments/evaluation/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "experiment_type": "ragas_comparison",
        "ragas_version": "0.4.3",
        "total_configs": len(configs),
        "successful_configs": len(ranked_results),
        "results": results
    }
    
    out_file = out_dir / "ragas_experiment.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n💾 详细结果保存: {out_file}")


if __name__ == "__main__":
    main()
