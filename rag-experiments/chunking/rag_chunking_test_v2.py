#!/usr/bin/env python3
"""
RAG文本分块 - LangChain版本
使用 langchain_text_splitters 替代自定义实现
"""
import json, os
from pathlib import Path
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter
)


def load_docs(path: str) -> list:
    """加载测试文档"""
    docs = []
    p = Path(path)
    for f in p.glob("*.md"):
        docs.append({"title": f.stem, "content": f.read_text(encoding="utf-8"), "type": "md"})
    json_dir = p / "mx_data" / "output"
    if json_dir.exists():
        for f in list(json_dir.glob("*_raw.json"))[:10]:
            docs.append({"title": f.stem, "content": json.dumps(json.load(f.open()), ensure_ascii=False), "type": "json"})
    return docs


def test_splitters(docs: list):
    """测试LangChain分块器"""
    text = "\n\n".join(f"# {d['title']}\n{d['content']}" for d in docs)
    
    splitters = {
        "fixed": CharacterTextSplitter(separator="\n\n", chunk_size=400, chunk_overlap=50),
        "recursive": RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50),
        "markdown": MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]),
    }
    
    results = []
    for name, splitter in splitters.items():
        chunks = splitter.split_text(text) if name != "markdown" else splitter.split_text(text)
        sizes = [len(c) for c in chunks]
        results.append({
            "strategy": name,
            "chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min": min(sizes),
            "max": max(sizes),
        })
        print(f"  {name}: {len(chunks)}块, 平均{sum(sizes)/len(sizes):.0f}字符")
    
    return results


def demo():
    print("=" * 50)
    print("RAG Chunking - LangChain")
    print("=" * 50)
    
    path = os.path.expanduser("~/Desktop/agent_demo")
    docs = load_docs(path)
    print(f"\n加载 {len(docs)} 个文档")
    
    results = test_splitters(docs)
    
    # 保存
    out = Path(path) / "chunking_test_results_v2.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n结果保存: {out}")


if __name__ == "__main__":
    demo()
