#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 三维评估体系

面试考点: Q6(评估Agent智能水平), Q9(量化评估)
核心思想: 效能 + 质量 + 鲁棒性 三维评估
"""

import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """指标类型"""
    EFFICIENCY = "efficiency"  # 效能
    QUALITY = "quality"        # 质量
    ROBUSTNESS = "robustness"  # 鲁棒性


@dataclass
class EvaluationResult:
    """评估结果"""
    task_id: str
    task_description: str
    success: bool
    steps_taken: int
    tokens_used: int
    execution_time_ms: float
    accuracy_score: float  # 0-1
    user_satisfaction: Optional[float] = None  # 0-5
    error_encountered: Optional[str] = None
    self_healed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "success": self.success,
            "steps_taken": self.steps_taken,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "accuracy_score": self.accuracy_score,
            "user_satisfaction": self.user_satisfaction,
            "error_encountered": self.error_encountered,
            "self_healed": self.self_healed
        }


class AgentEvaluator:
    """
    Agent 三维评估器
    
    面试要点:
    1. 效能维度: 完成率、步数、耗时、Token成本
    2. 质量维度: 准确率、满意度、一次解决率
    3. 鲁棒性维度: 异常处理成功率、自修复率、MTBF
    """
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.failure_categories: Dict[str, int] = {
            "intent_error": 0,      # 意图理解错误
            "planning_error": 0,     # 规划错误
            "tool_error": 0,         # 工具调用错误
            "memory_error": 0,       # 记忆错误
            "other": 0
        }
    
    def record(self, result: EvaluationResult):
        """记录评估结果"""
        self.results.append(result)
        
        # 分类失败原因
        if not result.success and result.error_encountered:
            self._categorize_failure(result.error_encountered)
    
    def _categorize_failure(self, error: str):
        """分类失败原因"""
        error_lower = error.lower()
        if any(kw in error_lower for kw in ["intent", "理解", "意图"]):
            self.failure_categories["intent_error"] += 1
        elif any(kw in error_lower for kw in ["plan", "规划", "步骤"]):
            self.failure_categories["planning_error"] += 1
        elif any(kw in error_lower for kw in ["tool", "工具", "调用"]):
            self.failure_categories["tool_error"] += 1
        elif any(kw in error_lower for kw in ["memory", "记忆", "上下文"]):
            self.failure_categories["memory_error"] += 1
        else:
            self.failure_categories["other"] += 1
    
    # ============ 效能维度 ============
    
    def efficiency_metrics(self) -> Dict:
        """效能指标"""
        if not self.results:
            return {}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        
        return {
            "task_completion_rate": successful / total,
            "avg_steps": sum(r.steps_taken for r in self.results) / total,
            "avg_execution_time_ms": sum(r.execution_time_ms for r in self.results) / total,
            "avg_tokens_per_task": sum(r.tokens_used for r in self.results) / total,
            "total_tokens": sum(r.tokens_used for r in self.results),
            "cost_estimate_usd": sum(r.tokens_used for r in self.results) * 0.000002  # $2/M tokens
        }
    
    # ============ 质量维度 ============
    
    def quality_metrics(self) -> Dict:
        """质量指标"""
        if not self.results:
            return {}
        
        total = len(self.results)
        successful = [r for r in self.results if r.success]
        
        metrics = {
            "accuracy": sum(r.accuracy_score for r in self.results) / total,
            "one_shot_resolution": len(successful) / total if successful else 0,
        }
        
        # 用户满意度
        satisfaction_scores = [r.user_satisfaction for r in self.results 
                              if r.user_satisfaction is not None]
        if satisfaction_scores:
            metrics["avg_user_satisfaction"] = sum(satisfaction_scores) / len(satisfaction_scores)
        
        return metrics
    
    # ============ 鲁棒性维度 ============
    
    def robustness_metrics(self) -> Dict:
        """鲁棒性指标"""
        if not self.results:
            return {}
        
        total = len(self.results)
        errors = [r for r in self.results if r.error_encountered]
        self_healed = [r for r in self.results if r.self_healed]
        
        return {
            "error_rate": len(errors) / total,
            "self_heal_rate": len(self_healed) / len(errors) if errors else 0,
            "mtbf_tasks": total / len(errors) if errors else float('inf'),  # Mean Time Between Failures
            "failure_breakdown": self.failure_categories
        }
    
    # ============ 综合报告 ============
    
    def generate_dashboard(self) -> Dict:
        """生成三维看板"""
        return {
            "efficiency": self.efficiency_metrics(),
            "quality": self.quality_metrics(),
            "robustness": self.robustness_metrics(),
            "summary": {
                "total_tasks": len(self.results),
                "overall_success_rate": sum(1 for r in self.results if r.success) / len(self.results) if self.results else 0,
                "evaluation_timestamp": time.time()
            }
        }
    
    def print_dashboard(self):
        """打印看板"""
        dashboard = self.generate_dashboard()
        
        print("\n" + "=" * 60)
        print("📊 Agent 三维评估看板")
        print("=" * 60)
        
        # 概览
        summary = dashboard["summary"]
        print(f"\n📋 概览:")
        print(f"  总任务数: {summary['total_tasks']}")
        print(f"  整体成功率: {summary['overall_success_rate']:.1%}")
        
        # 效能
        print(f"\n⚡ 效能维度:")
        eff = dashboard["efficiency"]
        for k, v in eff.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        
        # 质量
        print(f"\n✨ 质量维度:")
        qual = dashboard["quality"]
        for k, v in qual.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        
        # 鲁棒性
        print(f"\n🛡️  鲁棒性维度:")
        rob = dashboard["robustness"]
        for k, v in rob.items():
            if k == "failure_breakdown":
                print(f"  失败分类:")
                for fk, fv in v.items():
                    print(f"    {fk}: {fv}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")


# ============ 演示 ============

def demo():
    """演示三维评估"""
    
    evaluator = AgentEvaluator()
    
    # 模拟一批任务结果
    test_results = [
        # 成功案例
        EvaluationResult("T1", "查询股票", True, 2, 150, 1200, 0.95, 4.5),
        EvaluationResult("T2", "分析ROE", True, 3, 280, 2500, 0.90, 4.0),
        EvaluationResult("T3", "板块情绪", True, 2, 200, 1800, 0.88, 4.2),
        EvaluationResult("T4", "生成报告", True, 5, 450, 3500, 0.92, 4.8),
        
        # 失败案例 - 意图错误
        EvaluationResult("T5", "模糊查询", False, 1, 100, 800, 0.30, 2.0, 
                        error_encountered="意图理解错误: 用户输入不清晰"),
        
        # 失败案例 - 工具错误（自修复成功）
        EvaluationResult("T6", "计算指标", True, 4, 350, 3000, 0.85, 4.0,
                        error_encountered="工具参数错误", self_healed=True),
        
        # 失败案例 - 规划错误
        EvaluationResult("T7", "复杂分析", False, 8, 600, 5000, 0.50, 3.0,
                        error_encountered="规划错误: 步骤过多导致上下文溢出"),
        
        # 成功案例
        EvaluationResult("T8", "回测策略", True, 6, 500, 4200, 0.87, 4.3),
        EvaluationResult("T9", "风险评估", True, 3, 250, 2200, 0.93, 4.6),
        EvaluationResult("T10", "推荐股票", True, 4, 380, 3100, 0.89, 4.1),
    ]
    
    for result in test_results:
        evaluator.record(result)
    
    # 打印看板
    evaluator.print_dashboard()
    
    # 失败分析
    print("\n" + "=" * 60)
    print("🔍 失败根因分析:")
    print("=" * 60)
    
    failures = [r for r in evaluator.results if not r.success]
    for f in failures:
        print(f"\n  ❌ {f.task_id}: {f.task_description}")
        print(f"     错误: {f.error_encountered}")
        print(f"     建议: ", end="")
        if "intent" in f.error_encountered.lower():
            print("优化意图识别模块，增加澄清机制")
        elif "tool" in f.error_encountered.lower():
            print("加强工具参数校验，增加重试逻辑")
        elif "plan" in f.error_encountered.lower():
            print("优化规划策略，限制最大步数")
        else:
            print("需要进一步分析")


if __name__ == "__main__":
    demo()
