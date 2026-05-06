#!/usr/bin/env python3
"""
RAGAS评测实验 - 专业RAG评估框架
使用 ragas 0.4.x 进行多维度评估

RAGAS核心指标:
- faithfulness: 忠实度 (回答是否基于检索上下文)
- answer_relevancy: 回答相关性
- context_precision: 上下文精确率
- context_recall: 上下文召回率
- context_entity_recall: 实体召回率
- answer_similarity: 回答相似度
- answer_correctness: 回答正确性
"""
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# RAGAS imports (0.4.x API)
from ragas import evaluate
# RAGAS 0.4.x 指标导入（兼容未来版本，优先使用新路径避免弃用警告）
try:
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness,
    )
except ImportError:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness,
    )
from ragas.dataset_schema import SingleTurnSample

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


@dataclass
class QAPair:
    """评测问答对"""
    question: str
    answer: str
    ground_truth: str
    contexts: List[str] = field(default_factory=list)
    category: str = "general"


class RagasEvaluator:
    """RAGAS评测器 - 专业RAG评估"""
    
    def __init__(self, llm_model: str = "gpt-4o", embedding_model: str = "bge"):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.results: List[Dict] = []
        
        # 初始化LLM
        self.llm = None
        self.llm_ok = False
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=api_key)
                self.llm_ok = True
            else:
                print("⚠️ 未设置 OPENAI_API_KEY 环境变量")
        except Exception as e:
            print(f"⚠️ LLM初始化失败: {e}")
        
        # 初始化Embedding
        try:
            if embedding_model == "bge":
                self.embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
            else:
                self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
            self.embed_ok = True
        except Exception as e:
            print(f"⚠️ Embedding初始化失败: {e}")
            self.embedder = None
            self.embed_ok = False
    
    def build_rag(self, documents: List[Dict]) -> Chroma:
        """构建向量存储"""
        texts = [d["text"] for d in documents]
        metadatas = [{"id": d.get("id", f"doc_{i}")} for i, d in enumerate(documents)]
        
        store = Chroma.from_texts(
            texts=texts,
            embedding=self.embedder,
            metadatas=metadatas
        )
        return store
    
    def retrieve_contexts(self, store: Chroma, query: str, k: int = 5) -> List[str]:
        """检索上下文"""
        retriever = store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return [d.page_content for d in docs]
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """生成回答"""
        if not self.llm_ok:
            return "[LLM不可用]"
        
        context_text = "\n\n".join(contexts[:3])
        prompt = f"""基于以下上下文回答问题。如果上下文不包含答案，请明确说明。

上下文:
{context_text}

问题: {query}

回答:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"[生成错误: {e}]"
    
    def evaluate_single(self, qa: QAPair) -> Dict:
        """单条评测"""
        # 构建RAGAS输入
        sample = SingleTurnSample(
            user_input=qa.question,
            response=qa.answer,
            reference=qa.ground_truth,
            retrieved_contexts=qa.contexts
        )
        
        # 计算各指标（LLM未配置时跳过，避免报错）
        metrics_result = {}
        
        # 忠实度: 回答是否基于上下文
        try:
            metrics_result["faithfulness"] = faithfulness.single_turn_score(sample) if self.llm_ok else None
        except Exception as e:
            metrics_result["faithfulness"] = None
            if self.llm_ok:
                print(f"  faithfulness计算失败: {e}")
        
        # 回答相关性
        try:
            metrics_result["answer_relevancy"] = answer_relevancy.single_turn_score(sample) if self.llm_ok else None
        except Exception as e:
            metrics_result["answer_relevancy"] = None
        
        # 上下文精确率
        try:
            metrics_result["context_precision"] = context_precision.single_turn_score(sample) if self.llm_ok else None
        except Exception as e:
            metrics_result["context_precision"] = None
        
        # 上下文召回率
        try:
            metrics_result["context_recall"] = context_recall.single_turn_score(sample) if self.llm_ok else None
        except Exception as e:
            metrics_result["context_recall"] = None
        
        # 实体召回率
        try:
            metrics_result["context_entity_recall"] = context_entity_recall.single_turn_score(sample) if self.llm_ok else None
        except Exception as e:
            metrics_result["context_entity_recall"] = None
        
        # 回答相似度
        try:
            metrics_result["answer_similarity"] = answer_similarity.single_turn_score(sample) if self.llm_ok else None
        except Exception as e:
            metrics_result["answer_similarity"] = None
        
        # 回答正确性
        try:
            metrics_result["answer_correctness"] = answer_correctness.single_turn_score(sample) if self.llm_ok else None
        except Exception as e:
            metrics_result["answer_correctness"] = None
        
        return {
            "question": qa.question,
            "answer": qa.answer[:200],
            "ground_truth": qa.ground_truth[:200],
            "category": qa.category,
            **metrics_result
        }
    
    def evaluate_batch(self, qa_pairs: List[QAPair]) -> List[Dict]:
        """批量评测"""
        results = []
        
        print(f"\n开始评测 {len(qa_pairs)} 条数据...")
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\n[{i}/{len(qa_pairs)}] {qa.category}: {qa.question[:50]}...")
            
            start = time.time()
            result = self.evaluate_single(qa)
            result["eval_time_ms"] = (time.time() - start) * 1000
            
            # 打印关键指标
            valid_scores = {k: v for k, v in result.items() 
                          if k not in ["question", "answer", "ground_truth", "category", "eval_time_ms"] 
                          and v is not None}
            if valid_scores:
                avg_score = sum(valid_scores.values()) / len(valid_scores)
                print(f"  平均得分: {avg_score:.3f}")
                for k, v in valid_scores.items():
                    print(f"    {k}: {v:.3f}")
            
            results.append(result)
        
        return results
    
    def summarize(self, results: List[Dict]) -> Dict:
        """汇总统计"""
        if not results:
            return {}
        
        metrics = ["faithfulness", "answer_relevancy", "context_precision", 
                   "context_recall", "context_entity_recall", 
                   "answer_similarity", "answer_correctness"]
        
        summary = {}
        
        # 整体指标
        for metric in metrics:
            scores = [r[metric] for r in results if r.get(metric) is not None]
            if scores:
                summary[metric] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # 按类别统计
        categories = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        for cat, cat_results in categories.items():
            cat_summary = {}
            for metric in metrics:
                scores = [r[metric] for r in cat_results if r.get(metric) is not None]
                if scores:
                    cat_summary[metric] = sum(scores) / len(scores)
            categories[cat] = cat_summary
        
        summary["by_category"] = categories
        summary["total"] = len(results)
        summary["avg_eval_time_ms"] = sum(r.get("eval_time_ms", 0) for r in results) / len(results)
        
        return summary
    
    def print_report(self, summary: Dict):
        """打印评测报告"""
        print("\n" + "=" * 60)
        print("📊 RAGAS 评测报告")
        print("=" * 60)
        
        print(f"\n总样本数: {summary.get('total', 0)}")
        print(f"平均评测耗时: {summary.get('avg_eval_time_ms', 0):.0f}ms")
        
        # 核心指标
        print("\n【核心指标】")
        core_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        has_core = False
        for metric in core_metrics:
            if metric in summary:
                has_core = True
                m = summary[metric]
                print(f"  {metric:20s}: {m['mean']:.3f} (范围: {m['min']:.3f}-{m['max']:.3f}, n={m['count']})")
        if not has_core:
            print("  (LLM未配置，指标未计算。设置 OPENAI_API_KEY 后重试)")
        
        # 扩展指标
        print("\n【扩展指标】")
        ext_metrics = ["context_entity_recall", "answer_similarity", "answer_correctness"]
        has_ext = False
        for metric in ext_metrics:
            if metric in summary:
                has_ext = True
                m = summary[metric]
                print(f"  {metric:20s}: {m['mean']:.3f} (范围: {m['min']:.3f}-{m['max']:.3f}, n={m['count']})")
        if not has_ext:
            print("  (LLM未配置，指标未计算)")
        
        # 按类别
        if "by_category" in summary and summary["by_category"]:
            print("\n【按类别统计】")
            for cat, metrics in summary["by_category"].items():
                if metrics:
                    avg = sum(metrics.values()) / len(metrics)
                    print(f"  {cat:15s}: 平均 {avg:.3f}")
                    for k, v in metrics.items():
                        print(f"    {k}: {v:.3f}")
        
        # 综合评分
        all_scores = []
        for metric in core_metrics:
            if metric in summary:
                all_scores.append(summary[metric]["mean"])
        if all_scores:
            overall = sum(all_scores) / len(all_scores)
            print(f"\n【综合评分】{overall:.3f} / 1.0")
            if overall >= 0.8:
                print("  ✅ 优秀 - RAG系统表现良好")
            elif overall >= 0.6:
                print("  ⚠️  一般 - 有优化空间")
            else:
                print("  ❌ 较差 - 需要重大改进")
        else:
            print("\n【综合评分】N/A (LLM未配置)")
            print("  设置 OPENAI_API_KEY 环境变量后运行完整评测")


def create_test_data() -> List[QAPair]:
    """创建测试数据"""
    return [
        QAPair(
            question="沪电股份2024年的ROE是多少？",
            answer="沪电股份2024年ROE为18.5%，在PCB行业中处于领先水平。",
            ground_truth="沪电股份2024年ROE为18.5%。",
            contexts=[
                "沪电股份（002463）2024年财务报告显示，公司净资产收益率（ROE）达到18.5%，",
                "较上年提升2.3个百分点。作为PCB行业龙头企业，沪电股份在高端PCB领域",
                "保持技术领先优势，2024年实现营业收入XX亿元，同比增长XX%。"
            ],
            category="财务"
        ),
        QAPair(
            question="泰豪科技2025年净利润预计多少？",
            answer="根据最新财报，泰豪科技2025年预计净利润为XX亿元。",
            ground_truth="泰豪科技2025年预计净利润XX亿元。",
            contexts=[
                "泰豪科技2025年一季度报告显示，公司实现营业收入XX亿元，",
                "归属于上市公司股东的净利润XX亿元，同比增长XX%。"
            ],
            category="年报"
        ),
        QAPair(
            question="芯瑞达最近主力资金流向如何？",
            answer="芯瑞达近期主力资金呈净流入状态，近5日净流入约XX万元。",
            ground_truth="芯瑞达近5日主力资金净流入XX万元。",
            contexts=[
                "芯瑞达（002983）资金流向数据显示，近5日主力资金净流入XX万元，",
                "其中大单净流入XX万元，中单净流出XX万元。"
            ],
            category="资金"
        ),
        QAPair(
            question="半导体板块最近的涨停传导路径是什么？",
            answer="半导体板块涨停传导路径为：设备→材料→设计→封测。",
            ground_truth="半导体板块涨停传导路径：设备→材料→设计→封测。",
            contexts=[
                "2025年5月半导体板块异动分析：早盘半导体设备龙头北方华创涨停，",
                "随后传导至材料端（沪硅产业），午后设计端（韦尔股份）跟涨，",
                "尾盘封测端（长电科技）补涨。"
            ],
            category="分析"
        ),
        QAPair(
            question="贝贝虾评分系统包含哪些维度？",
            answer="贝贝虾评分系统包含市场环境、板块情绪、个股交易结构三个维度。",
            ground_truth="贝贝虾评分包含市场环境、板块情绪、个股交易结构三个维度。",
            contexts=[
                "贝贝虾板块情绪分析工具从三个维度进行综合打分：",
                "1. 市场环境：大盘走势、成交量、涨跌比",
                "2. 板块情绪：涨停数、龙头状态、资金流入",
                "3. 个股交易结构：技术面、基本面、资金流向"
            ],
            category="工具"
        ),
    ]


def demo():
    """演示RAGAS评测"""
    print("=" * 60)
    print("RAGAS RAG评测实验")
    print("=" * 60)
    
    # 初始化评测器
    evaluator = RagasEvaluator(llm_model="gpt-4o", embedding_model="bge")
    
    if not evaluator.llm_ok:
        print("\n⚠️ LLM未配置，请设置OPENAI_API_KEY环境变量")
        print("export OPENAI_API_KEY=your_key")
        return
    
    # 加载测试数据
    qa_pairs = create_test_data()
    print(f"\n加载 {len(qa_pairs)} 条测试数据")
    
    # 执行评测
    results = evaluator.evaluate_batch(qa_pairs)
    
    # 汇总
    summary = evaluator.summarize(results)
    evaluator.print_report(summary)
    
    # 保存结果
    out_dir = Path(os.path.expanduser("~/Desktop/agent_demo/rag-experiments/evaluation/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "ragas_version": "0.4.3",
        "llm_model": evaluator.llm_model,
        "embedding_model": evaluator.embedding_model,
        "summary": summary,
        "details": results
    }
    
    out_file = out_dir / "ragas_evaluation.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n💾 结果保存: {out_file}")
    
    # 生成对比报告
    print("\n" + "=" * 60)
    print("📊 与LangChain评估对比")
    print("=" * 60)
    print("""
RAGAS优势:
- 更细粒度的指标拆解 (7个维度 vs 2个维度)
- 自动化的LLM评判，减少人工标注
- 支持上下文级别的精确率和召回率
- 实体级别的召回分析

LangChain评估优势:
- 更轻量，无需额外依赖
- 可自定义评估标准
- 与LangSmith集成，便于追踪

建议:
- 开发阶段: 使用RAGAS进行深度分析
- 生产监控: 使用LangChain + LangSmith持续追踪
- 两者互补，形成完整评估体系
""")


if __name__ == "__main__":
    demo()
