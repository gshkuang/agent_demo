#!/usr/bin/env python3
"""
Agent三维评估 - LangSmith + CriteriaEvalChain
面试考点: Q6(评估Agent), Q9(量化评估)
"""
import json, time
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from langsmith import Client as LangSmithClient
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_openai import ChatOpenAI


class MetricType(Enum):
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    ROBUSTNESS = "robustness"


@dataclass
class EvalResult:
    task_id: str
    task_desc: str
    success: bool
    steps: int
    tokens: int
    time_ms: float
    accuracy: float
    error: Optional[str] = None
    healed: bool = False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class AgentEvaluator:
    """三维评估器 (LangChain/LangSmith)"""

    def __init__(self, project: str = "agent-eval"):
        self.results: List[EvalResult] = []
        self.failures = {"intent": 0, "planning": 0, "tool": 0, "memory": 0, "other": 0}
        try:
            self.langsmith = LangSmithClient()
            self.ls_ok = True
        except:
            self.langsmith = None
            self.ls_ok = False
        try:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
            self.evaluator = load_evaluator(EvaluatorType.CRITERIA,
                criteria={"accuracy": "回答准确？", "completeness": "回答完整？", "helpfulness": "有帮助？"},
                llm=self.llm)
            self.eval_ok = True
        except:
            self.evaluator = None
            self.eval_ok = False

    def record(self, r: EvalResult):
        self.results.append(r)
        if not r.success and r.error:
            self._classify(r.error)
        if self.ls_ok:
            try:
                # LangSmith反馈通过外部run_id注入
                pass
            except:
                pass

    def _classify(self, error: str):
        e = error.lower()
        if "intent" in e or "理解" in e: self.failures["intent"] += 1
        elif "plan" in e or "规划" in e: self.failures["planning"] += 1
        elif "tool" in e or "工具" in e: self.failures["tool"] += 1
        elif "memory" in e or "记忆" in e: self.failures["memory"] += 1
        else: self.failures["other"] += 1

    def evaluate(self, query: str, response: str, reference: str = None) -> Dict:
        """LangChain评估器评分"""
        if not self.eval_ok:
            return {"error": "评估器不可用"}
        try:
            if reference:
                ev = load_evaluator(EvaluatorType.QA, llm=self.llm)
                return ev.evaluate_strings(prediction=response, reference=reference, input=query)
            return self.evaluator.evaluate_strings(prediction=response, input=query)
        except Exception as e:
            return {"error": str(e)}

    def efficiency(self) -> Dict:
        if not self.results: return {}
        n = len(self.results)
        ok = sum(1 for r in self.results if r.success)
        return {
            "completion_rate": ok / n,
            "avg_steps": sum(r.steps for r in self.results) / n,
            "avg_time_ms": sum(r.time_ms for r in self.results) / n,
            "avg_tokens": sum(r.tokens for r in self.results) / n,
            "cost_usd": sum(r.tokens for r in self.results) * 2e-6
        }

    def quality(self) -> Dict:
        if not self.results: return {}
        n = len(self.results)
        return {
            "accuracy": sum(r.accuracy for r in self.results) / n,
            "one_shot": sum(1 for r in self.results if r.success) / n
        }

    def robustness(self) -> Dict:
        if not self.results: return {}
        n = len(self.results)
        errs = [r for r in self.results if r.error]
        healed = [r for r in self.results if r.healed]
        return {
            "error_rate": len(errs) / n,
            "heal_rate": len(healed) / len(errs) if errs else 0,
            "failures": self.failures
        }

    def dashboard(self) -> Dict:
        return {
            "efficiency": self.efficiency(),
            "quality": self.quality(),
            "robustness": self.robustness(),
            "summary": {"total": len(self.results), "success_rate": sum(1 for r in self.results if r.success) / len(self.results) if self.results else 0}
        }

    def print_dashboard(self):
        d = self.dashboard()
        print("\n" + "=" * 50)
        print("📊 Agent三维评估")
        print("=" * 50)
        print(f"总任务: {d['summary']['total']}, 成功率: {d['summary']['success_rate']:.1%}")
        for dim, metrics in [("⚡效能", d["efficiency"]), ("✨质量", d["quality"]), ("🛡️鲁棒", d["robustness"])]:
            print(f"\n{dim}:")
            for k, v in metrics.items():
                if isinstance(v, float): print(f"  {k}: {v:.3f}")
                elif isinstance(v, dict): print(f"  {k}: {v}")
                else: print(f"  {k}: {v}")


def demo():
    ev = AgentEvaluator()
    test_data = [
        EvalResult("T1", "查询股票", True, 2, 150, 1200, 0.95),
        EvalResult("T2", "分析ROE", True, 3, 280, 2500, 0.90),
        EvalResult("T3", "板块情绪", True, 2, 200, 1800, 0.88),
        EvalResult("T4", "生成报告", True, 5, 450, 3500, 0.92),
        EvalResult("T5", "模糊查询", False, 1, 100, 800, 0.30, "intent_error: 用户输入不清晰"),
        EvalResult("T6", "计算指标", True, 4, 350, 3000, 0.85, "tool_error: 参数错误", True),
    ]
    for r in test_data:
        ev.record(r)
    ev.print_dashboard()

    # LangChain评估器
    try:
        result = ev.evaluate("分析沪电股份", "ROE 18.5%，PCB龙头", "ROE 18.5%")
        print(f"\n评估分数: {result.get('score', 'N/A')}")
    except Exception as e:
        print(f"\n⚠️ 需要API Key: {e}")


if __name__ == "__main__":
    demo()
