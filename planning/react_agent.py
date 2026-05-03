#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReAct Agent 实现

面试考点: Q1(ReAct vs Plan-and-Execute), Q8(ReAct框架详解)
核心思想: Reasoning + Acting 交替循环
"""

import json
import re
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict
    func: Callable


@dataclass
class ThoughtAction:
    """ReAct 步骤"""
    thought: str
    action: str
    action_input: Dict
    observation: Optional[str] = None
    is_final: bool = False


class ReActAgent:
    """
    ReAct Agent: 推理与行动交替
    
    面试要点:
    - Thought: 分析当前状态，决定下一步
    - Action: 调用工具
    - Observation: 观察工具返回结果
    - Loop: 直到任务完成
    """
    
    def __init__(self, tools: List[Tool], max_steps: int = 10):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.history: List[ThoughtAction] = []
    
    def _build_prompt(self, query: str) -> str:
        """构建 ReAct Prompt"""
        tool_descs = "\n".join([
            f"{name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        history_str = ""
        for step in self.history:
            history_str += f"""
Thought: {step.thought}
Action: {step.action}
Action Input: {json.dumps(step.action_input, ensure_ascii=False)}
Observation: {step.observation}
"""
        
        prompt = f"""你是一个ReAct Agent，通过推理和行动交替完成任务。

可用工具:
{tool_descs}

请按以下格式思考:
Thought: 分析当前情况，决定下一步
Action: 工具名称
Action Input: {{"参数名": "值"}}
Observation: 工具返回结果（由系统自动填充）
... (重复直到任务完成)
Thought: 我现在知道最终答案
Final Answer: 最终答案

用户问题: {query}
{history_str}
Thought:"""
        return prompt
    
    def _parse_thought_action(self, text: str) -> ThoughtAction:
        """解析模型的 Thought + Action"""
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)', text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # 检查是否是最终答案
        final_match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
        if final_match:
            return ThoughtAction(
                thought=thought,
                action="final",
                action_input={},
                observation=final_match.group(1).strip(),
                is_final=True
            )
        
        # 提取 Action
        action_match = re.search(r'Action:\s*(\w+)', text)
        action = action_match.group(1) if action_match else ""
        
        # 提取 Action Input
        input_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL)
        action_input = {}
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except:
                action_input = {"raw": input_match.group(1)}
        
        return ThoughtAction(
            thought=thought,
            action=action,
            action_input=action_input
        )
    
    def run(self, query: str, llm_call: Callable[[str], str]) -> str:
        """
        执行 ReAct 循环
        
        面试要点: 必须设置步数上限，防止成本和时延失控
        """
        print(f"\n🎯 任务: {query}")
        print("=" * 60)
        
        for step in range(self.max_steps):
            # 1. 构建 Prompt
            prompt = self._build_prompt(query)
            
            # 2. LLM 生成 Thought + Action
            response = llm_call(prompt)
            step_data = self._parse_thought_action(response)
            
            print(f"\n📍 Step {step + 1}")
            print(f"🤔 Thought: {step_data.thought[:100]}...")
            
            # 3. 如果是最终答案，返回
            if step_data.is_final:
                print(f"✅ Final Answer: {step_data.observation[:150]}...")
                self.history.append(step_data)
                return step_data.observation
            
            # 4. 执行工具
            print(f"🔧 Action: {step_data.action}({step_data.action_input})")
            
            if step_data.action in self.tools:
                try:
                    tool = self.tools[step_data.action]
                    result = tool.func(**step_data.action_input)
                    step_data.observation = str(result)
                except Exception as e:
                    step_data.observation = f"错误: {str(e)}"
            else:
                step_data.observation = f"未知工具: {step_data.action}"
            
            print(f"👁️  Observation: {step_data.observation[:100]}...")
            self.history.append(step_data)
        
        print("⚠️ 达到最大步数限制")
        return "任务未完成，达到最大步数限制"


# ============ 模拟工具 ============

def search_stock(query: str) -> str:
    """搜索股票信息"""
    stocks = {
        "沪电股份": "代码002463，PCB龙头，2024年ROE 18.5%",
        "三安光电": "代码600703，LED芯片龙头，十大股东含香港中央结算",
        "芯瑞达": "代码002983，新型显示材料，近5日主力净流入1.2亿",
    }
    for name, info in stocks.items():
        if name in query:
            return info
    return f"未找到'{query}'的相关信息"


def calculate_roe(net_profit: float, equity: float) -> str:
    """计算ROE"""
    if equity <= 0:
        return "净资产必须大于0"
    roe = (net_profit / equity) * 100
    return f"ROE = {roe:.2f}%"


def get_market_sentiment() -> str:
    """获取市场情绪"""
    return "今日涨停45家，跌停12家，情绪偏乐观"


# ============ 模拟 LLM ============

class MockLLM:
    """模拟LLM，根据规则生成ReAct步骤"""
    
    def __init__(self):
        self.step = 0
        self.scenario = []
    
    def set_scenario(self, scenario: List[Dict]):
        """预设场景步骤"""
        self.scenario = scenario
        self.step = 0
    
    def __call__(self, prompt: str) -> str:
        if self.step < len(self.scenario):
            result = self.scenario[self.step]
            self.step += 1
            return result
        
        # 默认结束
        return "Thought: 我已经收集到足够信息\nFinal Answer: 分析完成"


def demo():
    """ReAct Agent 演示"""
    
    # 定义工具
    tools = [
        Tool("search_stock", "搜索股票基本信息", {"query": "str"}, search_stock),
        Tool("calculate_roe", "计算ROE", {"net_profit": "float", "equity": "float"}, calculate_roe),
        Tool("get_market_sentiment", "获取市场情绪", {}, get_market_sentiment),
    ]
    
    # 创建Agent
    agent = ReActAgent(tools=tools, max_steps=5)
    
    # 预设场景: 分析沪电股份
    llm = MockLLM()
    llm.set_scenario([
        """Thought: 用户想分析沪电股份，我需要先搜索它的基本信息
Action: search_stock
Action Input: {"query": "沪电股份"}""",
        """Thought: 找到了沪电股份的基本信息，ROE是18.5%。让我获取市场情绪作为参考
Action: get_market_sentiment
Action Input: {}""",
        """Thought: 我已经获取了沪电股份的基本信息和市场情绪。沪电股份是PCB龙头，ROE 18.5%，市场情绪偏乐观。可以给出分析了。
Final Answer: 沪电股份(002463)是PCB行业龙头，2024年ROE为18.5%，高于行业平均水平。当前市场情绪偏乐观(涨停45家)，建议关注。"""
    ])
    
    result = agent.run("帮我分析沪电股份的投资价值", llm)
    
    print("\n" + "=" * 60)
    print("📊 执行统计:")
    print(f"  总步数: {len(agent.history)}")
    print(f"  工具调用: {sum(1 for s in agent.history if not s.is_final)}次")
    print(f"  最终结果: {result[:80]}...")


if __name__ == "__main__":
    demo()
