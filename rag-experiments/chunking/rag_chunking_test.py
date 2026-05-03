#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG文本分块测试脚本
针对掘金文章《切实有效的RAG文本分块》的方法论，测试不同分块策略对金融文档的检索效果
"""

import json
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple
from pathlib import Path

# ============== 数据模型 ==============

@dataclass
class Chunk:
    """文本块数据模型"""
    content: str
    source: str = ""           # 来源文档
    heading_path: List[str] = field(default_factory=list)  # 标题层级路径
    chunk_type: str = ""       # 块类型（paragraph/code/table/warning等）
    start_pos: int = 0         # 在原文中的起始位置
    end_pos: int = 0           # 在原文中的结束位置
    metadata: Dict = field(default_factory=dict)

@dataclass
class EvalCase:
    """评估用例"""
    question: str
    expected_keywords: List[str]   # 期望检索到的关键词
    category: str                  # 问题类别

@dataclass
class EvalResult:
    """评估结果"""
    strategy_name: str
    hit_rate: float
    avg_chunk_size: float
    num_chunks: int
    details: List[Dict]


# ============== 分块策略实现 ==============

class ChunkingStrategies:
    """四种分块策略实现"""
    
    @staticmethod
    def fixed_length(text: str, max_tokens: int = 400, overlap: int = 50) -> List[Chunk]:
        """固定长度分块 - 最基础的方式"""
        words = text.split()
        chunks = []
        i = 0
        chunk_idx = 0
        
        while i < len(words):
            end = min(i + max_tokens, len(words))
            chunk_words = words[i:end]
            chunk_text = " ".join(chunk_words)
            
            # 计算字符位置
            start_char = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            chunks.append(Chunk(
                content=chunk_text,
                source="fixed_length",
                chunk_type="fixed",
                start_pos=start_char,
                end_pos=end_char,
                metadata={"chunk_index": chunk_idx, "word_count": len(chunk_words)}
            ))
            
            i += max_tokens - overlap
            chunk_idx += 1
            
        return chunks
    
    @staticmethod
    def recursive(text: str, max_tokens: int = 512) -> List[Chunk]:
        """递归分块 - 按标题→段落→句子逐级降级"""
        # 分隔符优先级：标题 > 段落 > 句子
        separators = [
            r"\n#{1,6}\s",      # Markdown标题
            r"\n\n",             # 段落
            r"(?<=[。！？\.\!\?])\s+",  # 句子结尾
        ]
        
        def _split_recursive(content: str, seps: List[str], max_t: int, 
                             heading_path: List[str] = None) -> List[Chunk]:
            if heading_path is None:
                heading_path = []
                
            words = content.split()
            if len(words) <= max_t or not seps:
                if content.strip():
                    return [Chunk(
                        content=content.strip(),
                        source="recursive",
                        heading_path=heading_path.copy(),
                        chunk_type="text",
                        metadata={"word_count": len(words)}
                    )]
                return []
            
            # 使用当前分隔符拆分
            parts = re.split(seps[0], content)
            chunks = []
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                part_words = part.split()
                
                # 检查是否是标题
                if seps[0] == r"\n#{1,6}\s" and part:
                    # 更新标题路径
                    heading_match = re.match(r"^(#{1,6})\s+(.+)$", part, re.MULTILINE)
                    if heading_match:
                        level = len(heading_match.group(1))
                        title = heading_match.group(2).strip()
                        # 维护标题层级栈
                        while heading_path and len(heading_path) >= level:
                            heading_path.pop()
                        heading_path.append(title)
                        continue  # 标题本身不作为内容块
                
                if len(part_words) <= max_t:
                    chunks.append(Chunk(
                        content=part,
                        source="recursive",
                        heading_path=heading_path.copy(),
                        chunk_type="paragraph",
                        metadata={"word_count": len(part_words)}
                    ))
                else:
                    # 递归使用下一级分隔符
                    chunks.extend(_split_recursive(part, seps[1:], max_t, heading_path.copy()))
            
            return chunks
        
        return _split_recursive(text, separators, max_tokens)
    
    @staticmethod
    def semantic(text: str, similarity_threshold: float = 0.7) -> List[Chunk]:
        """语义分块 - 基于主题变化检测（简化版，使用句子相似度）"""
        # 先按句子拆分
        sentences = re.split(r"(?<=[。！？\.\!\?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # 简化版：使用关键词重叠度作为相似度代理
        # 实际生产环境应使用嵌入向量
        def _simple_similarity(sent1: str, sent2: str) -> float:
            """基于关键词重叠的简化相似度计算"""
            words1 = set(re.findall(r'[\u4e00-\u9fff\w]+', sent1.lower()))
            words2 = set(re.findall(r'[\u4e00-\u9fff\w]+', sent2.lower()))
            if not words1 or not words2:
                return 0.0
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # 计算当前句子与上一句的相似度
            sim = _simple_similarity(sentences[i-1], sentences[i])
            
            if sim < similarity_threshold and current_chunk_sentences:
                # 主题切换，保存当前块
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    content=chunk_text,
                    source="semantic",
                    chunk_type="semantic_block",
                    metadata={
                        "sentence_count": len(current_chunk_sentences),
                        "word_count": len(chunk_text.split()),
                        "avg_similarity": sim
                    }
                ))
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])
        
        # 保存最后一个块
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(Chunk(
                content=chunk_text,
                source="semantic",
                chunk_type="semantic_block",
                metadata={
                    "sentence_count": len(current_chunk_sentences),
                    "word_count": len(chunk_text.split())
                }
            ))
        
        return chunks
    
    @staticmethod
    def structured_aware(text: str, max_tokens: int = 512) -> List[Chunk]:
        """结构化感知分块 - 针对Markdown/技术文档"""
        chunks = []
        
        # 识别代码块
        code_blocks = list(re.finditer(r"```[\s\S]*?```", text))
        code_ranges = [(m.start(), m.end()) for m in code_blocks]
        
        # 识别表格
        table_blocks = list(re.finditer(r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)", text))
        table_ranges = [(m.start(), m.end()) for m in table_blocks]
        
        # 识别警告/提示框
        warning_pattern = r"^>\s*\*\*(Warning|Note|Caution|警告|注意|提示)\*\*.*?(?=\n\n|\Z)"
        warning_blocks = list(re.finditer(warning_pattern, text, re.MULTILINE | re.DOTALL))
        warning_ranges = [(m.start(), m.end()) for m in warning_blocks]
        
        # 识别标题
        heading_pattern = r"^(#{1,6})\s+(.+)$"
        headings = list(re.finditer(heading_pattern, text, re.MULTILINE))
        
        # 构建标题路径
        heading_stack = []
        heading_positions = []
        for h in headings:
            level = len(h.group(1))
            title = h.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading_positions.append((h.start(), h.end(), [h[1] for h in heading_stack]))
        
        # 按段落拆分，但保留结构化元素
        paragraphs = re.split(r"\n\n", text)
        current_heading_path = []
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()
            if not para:
                i += 1
                continue
            
            # 更新当前标题路径
            for pos, end, path in heading_positions:
                if pos <= text.find(para) < end + 1:
                    current_heading_path = path
            
            # 检查是否是结构化元素
            is_code = any(start <= text.find(para) < end for start, end in code_ranges)
            is_table = any(start <= text.find(para) < end for start, end in table_ranges)
            is_warning = any(start <= text.find(para) < end for start, end in warning_ranges)
            
            chunk_type = "paragraph"
            if is_code:
                chunk_type = "code"
            elif is_table:
                chunk_type = "table"
            elif is_warning:
                chunk_type = "warning"
            elif para.startswith("#"):
                chunk_type = "heading"
            
            # 警告块需要与前文绑定
            if is_warning and i > 0 and chunks:
                # 将警告追加到前一个块
                chunks[-1].content += "\n\n" + para
                chunks[-1].metadata["has_warning"] = True
                chunks[-1].chunk_type = "paragraph_with_warning"
                i += 1
                continue
            
            chunks.append(Chunk(
                content=para,
                source="structured_aware",
                heading_path=current_heading_path.copy(),
                chunk_type=chunk_type,
                metadata={
                    "is_structured": is_code or is_table or is_warning,
                    "word_count": len(para.split())
                }
            ))
            i += 1
        
        return chunks


# ============== 评估系统 ==============

class RAGEvaluator:
    """RAG分块策略评估器"""
    
    def __init__(self, test_cases: List[EvalCase]):
        self.test_cases = test_cases
    
    def evaluate(self, chunks: List[Chunk], strategy_name: str) -> EvalResult:
        """评估分块策略"""
        details = []
        hits = 0
        
        for case in self.test_cases:
            # 模拟检索：检查关键词是否出现在Top-K块中
            found_keywords = set()
            for chunk in chunks:
                for kw in case.expected_keywords:
                    if kw in chunk.content:
                        found_keywords.add(kw)
            
            hit = len(found_keywords) == len(case.expected_keywords)
            if hit:
                hits += 1
            
            details.append({
                "question": case.question,
                "category": case.category,
                "found": list(found_keywords),
                "expected": case.expected_keywords,
                "hit": hit
            })
        
        hit_rate = hits / len(self.test_cases) if self.test_cases else 0
        avg_size = sum(len(c.content.split()) for c in chunks) / max(len(chunks), 1)
        
        return EvalResult(
            strategy_name=strategy_name,
            hit_rate=hit_rate,
            avg_chunk_size=avg_size,
            num_chunks=len(chunks),
            details=details
        )


# ============== 测试数据加载 ==============

def load_test_documents(demo_path: str) -> List[Dict]:
    """加载agent_demo中的测试文档"""
    docs = []
    demo_dir = Path(demo_path)
    
    if not demo_dir.exists():
        print(f"警告: {demo_path} 不存在")
        return docs
    
    # 加载Markdown分析文档
    for md_file in demo_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        docs.append({
            "title": md_file.stem,
            "content": content,
            "type": "analysis",
            "path": str(md_file)
        })
    
    # 加载JSON数据文件
    json_dir = demo_dir / "mx_data" / "output"
    if json_dir.exists():
        for json_file in json_dir.glob("*_raw.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 将JSON转为可读的文本格式
                content = json.dumps(data, ensure_ascii=False, indent=2)
                docs.append({
                    "title": json_file.stem,
                    "content": content,
                    "type": "financial_data",
                    "path": str(json_file)
                })
            except Exception as e:
                print(f"加载 {json_file} 失败: {e}")
    
    return docs


# ============== 主测试流程 ==============

def run_chunking_test():
    """运行分块测试"""
    print("=" * 80)
    print("RAG文本分块策略测试")
    print("测试数据来源: ~/Desktop/agent_demo")
    print("=" * 80)
    
    # 加载测试文档
    demo_path = os.path.expanduser("~/Desktop/agent_demo")
    docs = load_test_documents(demo_path)
    
    if not docs:
        print("未找到测试文档，请确认agent_demo文件夹存在且包含数据")
        return
    
    print(f"\n加载了 {len(docs)} 个测试文档:")
    for doc in docs:
        print(f"  - {doc['title']} ({doc['type']}, {len(doc['content'])} 字符)")
    
    # 构建评估用例（基于金融文档特点）
    eval_cases = [
        EvalCase(
            question="沪电股份的ROE和毛利率是多少？",
            expected_keywords=["沪电股份", "ROE", "毛利率"],
            category="财务指标查询"
        ),
        EvalCase(
            question="泰豪科技2025年的营业收入和净利润？",
            expected_keywords=["泰豪科技", "营业收入", "净利润"],
            category="年报数据查询"
        ),
        EvalCase(
            question="芯瑞达的主力资金流向情况？",
            expected_keywords=["芯瑞达", "主力资金"],
            category="资金流向查询"
        ),
        EvalCase(
            question="三安光电的十大流通股东有哪些？",
            expected_keywords=["三安光电", "流通股东"],
            category="股东信息查询"
        ),
        EvalCase(
            question="科瑞技术的市盈率和市净率？",
            expected_keywords=["科瑞技术", "市盈率", "市净率"],
            category="估值指标查询"
        ),
        EvalCase(
            question="板块轮动复盘中的涨停传导路径？",
            expected_keywords=["涨停", "传导", "板块"],
            category="分析文档查询"
        ),
        EvalCase(
            question="贝贝虾分析中的个股评分？",
            expected_keywords=["贝贝虾", "评分"],
            category="分析文档查询"
        ),
        EvalCase(
            question="回测系统的数据爬取逻辑？",
            expected_keywords=["回测", "爬取", "数据"],
            category="技术文档查询"
        ),
    ]
    
    # 合并所有文档内容
    all_content = "\n\n".join([f"# {doc['title']}\n{doc['content']}" for doc in docs])
    
    print(f"\n总文本长度: {len(all_content)} 字符, {len(all_content.split())} 词")
    print(f"评估用例数: {len(eval_cases)}")
    
    # 初始化评估器
    evaluator = RAGEvaluator(eval_cases)
    
    # 测试四种分块策略
    strategies = {
        "固定长度分块": lambda text: ChunkingStrategies.fixed_length(text, max_tokens=400, overlap=50),
        "递归分块": lambda text: ChunkingStrategies.recursive(text, max_tokens=512),
        "语义分块": lambda text: ChunkingStrategies.semantic(text, similarity_threshold=0.3),
        "结构化感知分块": lambda text: ChunkingStrategies.structured_aware(text, max_tokens=512),
    }
    
    results = []
    
    print("\n" + "=" * 80)
    print("开始测试各分块策略...")
    print("=" * 80)
    
    for name, strategy_fn in strategies.items():
        print(f"\n【{name}】")
        print("-" * 60)
        
        try:
            chunks = strategy_fn(all_content)
            result = evaluator.evaluate(chunks, name)
            results.append(result)
            
            print(f"  生成块数: {result.num_chunks}")
            print(f"  平均块大小: {result.avg_chunk_size:.1f} 词")
            print(f"  检索命中率: {result.hit_rate * 100:.1f}%")
            
            # 显示块大小分布
            sizes = [len(c.content.split()) for c in chunks]
            print(f"  块大小分布: min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes)//2]}")
            
            # 显示详细评估结果
            print(f"\n  详细评估:")
            for detail in result.details:
                status = "✓" if detail["hit"] else "✗"
                print(f"    {status} [{detail['category']}] {detail['question'][:40]}...")
                if not detail["hit"]:
                    print(f"      找到: {detail['found']}, 期望: {detail['expected']}")
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总对比
    print("\n" + "=" * 80)
    print("策略对比汇总")
    print("=" * 80)
    print(f"{'策略名称':<20} {'块数':>8} {'平均大小':>10} {'命中率':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.strategy_name:<20} {r.num_chunks:>8} {r.avg_chunk_size:>10.1f} {r.hit_rate*100:>9.1f}%")
    
    # 找出最佳策略
    if results:
        best = max(results, key=lambda x: x.hit_rate)
        print(f"\n最佳策略: {best.strategy_name} (命中率: {best.hit_rate*100:.1f}%)")
    
    # 保存详细结果
    output_path = os.path.expanduser("~/Desktop/agent_demo/chunking_test_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([{
            "strategy": r.strategy_name,
            "hit_rate": r.hit_rate,
            "num_chunks": r.num_chunks,
            "avg_chunk_size": r.avg_chunk_size,
            "details": r.details
        } for r in results], f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存至: {output_path}")
    
    return results


if __name__ == "__main__":
    run_chunking_test()
