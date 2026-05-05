#!/usr/bin/env python3
"""
RAG端到端评估 - LangChain版本
使用 load_evaluator + Chroma + 真实BGE模型
"""
import json, os, time
from pathlib import Path
from dataclasses import dataclass
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.prompts import PromptTemplate


@dataclass
class QAPair:
    question: str
    expected: str
    keywords: List[str]
    category: str


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


def build_rag(docs: list, embedder):
    """构建RAG系统"""
    store = Chroma.from_texts(
        texts=[d["text"] for d in docs],
        embedding=embedder,
        metadatas=[{"id": d["id"]} for d in docs]
    )
    retriever = store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return retriever, llm


def evaluate_rag(retriever, llm, qa_pairs: list):
    """评估RAG系统"""
    # 加载评估器
    try:
        faith_evaluator = load_evaluator(
            EvaluatorType.CRITERIA,
            criteria={"faithfulness": "回答是否基于提供的上下文？"},
            llm=llm
        )
        relevance_evaluator = load_evaluator(
            EvaluatorType.CRITERIA,
            criteria={"relevance": "回答是否相关？"},
            llm=llm
        )
    except:
        faith_evaluator = None
        relevance_evaluator = None
    
    results = []
    for qa in qa_pairs:
        # 检索
        start = time.time()
        chunks = retriever.invoke(qa.question)
        ret_latency = (time.time() - start) * 1000
        
        context = "\n\n".join([c.page_content for c in chunks[:3]])
        
        # 生成回答
        prompt = f"基于以下上下文回答问题:\n{context}\n\n问题: {qa.question}\n回答:"
        start = time.time()
        answer = llm.invoke(prompt).content
        gen_latency = (time.time() - start) * 1000
        
        # 评估
        faith = 0.0
        relevance = 0.0
        if faith_evaluator:
            try:
                faith_result = faith_evaluator.evaluate_strings(
                    prediction=answer,
                    input=qa.question,
                    reference=context
                )
                faith = faith_result.get("score", 0)
            except:
                pass
        
        if relevance_evaluator:
            try:
                rel_result = relevance_evaluator.evaluate_strings(
                    prediction=answer,
                    input=qa.question
                )
                relevance = rel_result.get("score", 0)
            except:
                pass
        
        # 关键词匹配
        keyword_hit = sum(1 for kw in qa.keywords if kw in answer) / len(qa.keywords)
        
        results.append({
            "question": qa.question,
            "answer": answer[:200],
            "faithfulness": faith,
            "relevance": relevance,
            "keyword_hit": keyword_hit,
            "retrieval_ms": ret_latency,
            "generation_ms": gen_latency,
        })
        
        print(f"  [{qa.category}] {qa.question[:40]}...")
        print(f"    忠实度: {faith:.2f}, 相关性: {relevance:.2f}, 关键词: {keyword_hit:.2f}")
    
    return results


def demo():
    print("=" * 50)
    print("RAG End-to-End Evaluation - LangChain")
    print("=" * 50)
    
    docs = load_docs(os.path.expanduser("~/Desktop/agent_demo"))
    print(f"\n加载 {len(docs)} 个文档")
    
    qa_pairs = [
        QAPair("沪电股份ROE是多少？", "ROE 18.5%", ["沪电", "ROE", "18"], "财务"),
        QAPair("泰豪科技2025年净利润？", "净利润XX亿", ["泰豪", "净利润"], "年报"),
        QAPair("芯瑞达主力资金流向？", "净流入XX万", ["芯瑞达", "资金"], "资金"),
        QAPair("板块轮动涨停传导路径？", "半导体→PCB", ["轮动", "涨停", "传导"], "分析"),
    ]
    
    # 使用BGE嵌入
    try:
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    except:
        print("BGE加载失败，使用OpenAI")
        embedder = OpenAIEmbeddings()
    
    retriever, llm = build_rag(docs, embedder)
    
    print("\n评估中...")
    results = evaluate_rag(retriever, llm, qa_pairs)
    
    # 汇总
    avg = {
        "faithfulness": sum(r["faithfulness"] for r in results) / len(results),
        "relevance": sum(r["relevance"] for r in results) / len(results),
        "keyword_hit": sum(r["keyword_hit"] for r in results) / len(results),
        "avg_latency_ms": sum(r["retrieval_ms"] + r["generation_ms"] for r in results) / len(results),
    }
    
    print(f"\n📊 汇总:")
    print(f"  忠实度: {avg['faithfulness']:.3f}")
    print(f"  相关性: {avg['relevance']:.3f}")
    print(f"  关键词: {avg['keyword_hit']:.3f}")
    print(f"  延迟: {avg['avg_latency_ms']:.0f}ms")
    
    # 保存
    out = Path(os.path.expanduser("~/Desktop/rag-experiments/evaluation/results"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2e_evaluation.json").write_text(json.dumps({"details": results, "summary": avg}, ensure_ascii=False, indent=2))
    print(f"\n结果保存: {out}/e2e_evaluation.json")


if __name__ == "__main__":
    demo()
