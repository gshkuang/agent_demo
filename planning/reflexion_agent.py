#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reflexion Agent 实现

面试考点: Q2(自我反思), Q6(Tree of Thoughts)
核心思想: 执行 → 反思 → 改进 的循环
"""

import json
from typing import List, Dict, Callable
from dataclasses import dataclass, field


@dataclass
class Reflection:
    """反思记录"""
    task: str
    attempt: int
    output: str
    critique: str
    improvement: str
    success: bool


class ReflexionAgent:
    """
    Reflexion Agent: 带自我反思的Agent
    
    面试要点:
    - 失败后不直接放弃，而是生成改进建议
    - 将经验教训存入记忆，下次避免同样错误
    - 需要设置停止条件，防止无限循环
    """
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.reflections: List[Reflection] = []
        self.memory: List[str] = []  # 经验教训
    
    def _build_prompt(self, task: str, context: str = "") -> str:
        """构建带反思的Prompt"""
        memory_str = "\n".join([
            f"- {m}" for m in self.memory[-5:]  # 最近5条经验
        ]) if self.memory else "暂无"
        
        prompt = f"""你是一个会自我反思的Agent。完成任务后，你需要评估结果并生成改进建议。

历史经验教训:
{memory_str}

{context}

任务: {task}

请直接输出结果，格式:
Result: <你的答案>
Confidence: <0-1之间的置信度>"""
        return prompt
    
    def _build_reflection_prompt(self, task: str, output: str) -> str:
        """构建反思Prompt"""
        prompt = f"""请对以下任务的输出进行批判性反思:

任务: {task}
输出: {output}

请分析:
1. 输出是否正确？有没有错误？
2. 有没有遗漏重要信息？
3. 如何改进？

格式:
Critique: <批评意见>
Improvement: <改进建议>
Success: <true/false>"""
        return prompt
    
    def execute(self, task: str, llm_call: Callable[[str], str]) -> Dict:
        """
        执行带反思的任务
        
        面试要点: 重度的ToT不能在线上用，但轻量化版本是杀手锏
        """
        print(f"\n🎯 任务: {task}")
        print("=" * 60)
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n🔄 尝试 {attempt}/{self.max_attempts}")
            
            # 1. 执行任务
            context = f"这是第{attempt}次尝试。之前的尝试都失败了，请特别注意。" if attempt > 1 else ""
            prompt = self._build_prompt(task, context)
            response = llm_call(prompt)
            
            # 解析结果
            result = self._parse_result(response)
            print(f"📤 输出: {result['result'][:80]}...")
            print(f"📊 置信度: {result.get('confidence', 'N/A')}")
            
            # 2. 反思
            reflection_prompt = self._build_reflection_prompt(task, result['result'])
            reflection_response = llm_call(reflection_prompt)
            reflection = self._parse_reflection(reflection_response)
            
            print(f"🤔 反思: {reflection['critique'][:80]}...")
            print(f"💡 改进: {reflection['improvement'][:80]}...")
            
            # 记录反思
            ref = Reflection(
                task=task,
                attempt=attempt,
                output=result['result'],
                critique=reflection['critique'],
                improvement=reflection['improvement'],
                success=reflection['success']
            )
            self.reflections.append(ref)
            
            # 如果成功，返回结果
            if reflection['success']:
                print(f"✅ 任务成功完成!")
                self.memory.append(f"任务'{task}'的成功经验: {reflection['improvement']}")
                return {
                    "success": True,
                    "result": result['result'],
                    "attempts": attempt,
                    "reflections": self.reflections
                }
            
            # 记录失败经验
            self.memory.append(f"任务'{task}'的失败教训: {reflection['critique']}")
            print(f"❌ 尝试失败，准备改进...")
        
        # 所有尝试都失败
        print(f"⚠️ 所有{self.max_attempts}次尝试均失败")
        return {
            "success": False,
            "result": result['result'],
            "attempts": self.max_attempts,
            "reflections": self.reflections
        }
    
    def _parse_result(self, text: str) -> Dict:
        """解析结果"""
        result = {"result": "", "confidence": 0.5}
        
        result_match = text.split("Result:")
        if len(result_match) > 1:
            result['result'] = result_match[1].split("Confidence:")[0].strip()
        
        conf_match = text.split("Confidence:")
        if len(conf_match) > 1:
            try:
                result['confidence'] = float(conf_match[1].strip().split()[0])
            except:
                pass
        
        return result
    
    def _parse_reflection(self, text: str) -> Dict:
        """解析反思"""
        reflection = {"critique": "", "improvement": "", "success": False}
        
        critique_match = text.split("Critique:")
        if len(critique_match) > 1:
            reflection['critique'] = critique_match[1].split("Improvement:")[0].strip()
        
        improvement_match = text.split("Improvement:")
        if len(improvement_match) > 1:
            reflection['improvement'] = improvement_match[1].split("Success:")[0].strip()
        
        success_match = text.split("Success:")
        if len(success_match) > 1:
            reflection['success'] = "true" in success_match[1].lower()
        
        return reflection
    
    def get_learnings(self) -> List[str]:
        """获取学习到的经验"""
        return self.memory


# ============ 模拟 LLM ============

class MockLLM:
    """模拟LLM，模拟反思过程"""
    
    def __init__(self):
        self.call_count = 0
        self.scenarios = {}
    
    def set_scenario(self, task: str, attempts: List[Dict]):
        """预设场景"""
        self.scenarios[task] = attempts
        self.call_count = 0
    
    def __call__(self, prompt: str) -> str:
        # 根据prompt内容返回不同结果
        if "Critique:" in prompt or "反思" in prompt:
            # 反思请求
            return self._generate_reflection(prompt)
        else:
            # 执行任务
            return self._generate_result(prompt)
    
    def _generate_result(self, prompt: str) -> str:
        """模拟任务执行"""
        if "ROE" in prompt:
            if "第2次" in prompt:
                return "Result: 沪电股份ROE = 18.5%，计算方式: 净利润/净资产 = 8.5/45.9 = 18.5%\nConfidence: 0.95"
            else:
                return "Result: 沪电股份ROE = 15.2%\nConfidence: 0.6"
        return "Result: 分析完成\nConfidence: 0.8"
    
    def _generate_reflection(self, prompt: str) -> str:
        """模拟反思"""
        if "15.2%" in prompt:
            return """Critique: ROE计算错误。之前用的是去年的数据，应该用2024年年报的最新数据。
Improvement: 使用最新年报数据重新计算，确认净利润和净资产的数值。
Success: false"""
        elif "18.5%" in prompt:
            return """Critique: 计算正确，数据来源清晰。
Improvement: 可以补充行业对比数据。
Success: true"""
        return """Critique: 输出基本正确。
Improvement: 可以增加更多细节。
Success: true"""


def demo():
    """Reflexion Agent 演示"""
    
    agent = ReflexionAgent(max_attempts=3)
    llm = MockLLM()
    
    # 场景: 第一次计算错误，第二次纠正
    result = agent.execute("计算沪电股份的ROE", llm)
    
    print("\n" + "=" * 60)
    print("📊 执行统计:")
    print(f"  成功: {result['success']}")
    print(f"  尝试次数: {result['attempts']}")
    print(f"  最终结果: {result['result'][:80]}...")
    
    print("\n🧠 学习到的经验:")
    for i, learning in enumerate(agent.get_learnings(), 1):
        print(f"  {i}. {learning[:80]}...")
    
    # 第二次执行同样任务，应该更快成功
    print("\n" + "=" * 60)
    print("🔄 再次执行同样任务（利用经验）:")
    agent2 = ReflexionAgent(max_attempts=3)
    agent2.memory = agent.memory  # 传递经验
    
    result2 = agent2.execute("计算沪电股份的ROE", llm)
    print(f"  尝试次数: {result2['attempts']} (应该更少)")


if __name__ == "__main__":
    demo()
