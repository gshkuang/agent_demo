#!/usr/bin/env python3
"""
Google ADK (Agent Development Kit) 演示
P2: Google 官方 Agent 框架

特点:
- 原生支持 Google Gemini
- 内置工具集成（搜索、代码执行）
- 支持多 Agent 协作
- 与 Google Cloud 深度集成
"""
import os
import asyncio
from typing import Dict, List, Optional

try:
    from google.adk.agents import Agent
    from google.adk.tools import google_search, built_in_code_execution
    from google.adk.sessions import Session
    from google.adk.runners import Runner
except ImportError:
    print("请先安装 Google ADK: pip install google-adk")
    print("文档: https://google.github.io/adk-docs/")
    raise


# ============ 配置 ============

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


# ============ 单 Agent 演示 ============

class StockAnalysisAgent:
    """股票分析 Agent - ADK 实现"""
    
    def __init__(self):
        self.agent = Agent(
            model="gemini-2.0-flash-exp",  # 或 gemini-2.0-pro-exp
            name="stock_analyzer",
            description="专业的股票分析助手",
            instruction="""你是专业的股票分析师，擅长基本面分析和技术分析。

你的分析流程:
1. 使用 google_search 搜索最新公司信息和财报
2. 分析财务指标（ROE、PE、PB、现金流）
3. 对比同行业公司
4. 给出投资建议

规则:
- 必须基于真实数据进行分析
- 给出明确的买入/持有/观望/卖出建议
- 说明投资逻辑和风险点
""",
            tools=[google_search],
        )
        self.runner = Runner(agent=self.agent)
    
    async def analyze(self, stock_name: str) -> str:
        """分析股票"""
        session = Session()
        
        result = await self.runner.run_async(
            session=session,
            query=f"请深度分析 {stock_name} 的投资价值，包括基本面、估值、行业地位",
        )
        
        return result.response


# ============ 多 Agent 协作 ============

class InvestmentTeam:
    """投研团队 - ADK 多 Agent"""
    
    def __init__(self):
        self._create_agents()
    
    def _create_agents(self):
        """创建团队成员"""
        
        # 数据收集 Agent
        self.data_agent = Agent(
            model="gemini-2.0-flash-exp",
            name="data_collector",
            description="数据收集员",
            instruction="""你负责收集公司的全面数据。
使用 google_search 搜索：
1. 最新财报数据
2. 行业新闻
3. 分析师评级
4. 竞争对手情况

输出结构化的数据摘要。""",
            tools=[google_search],
        )
        
        # 分析 Agent
        self.analysis_agent = Agent(
            model="gemini-2.0-flash-exp",
            name="analyst",
            description="投资分析师",
            instruction="""你基于收集的数据进行深度分析。
分析维度:
1. 财务健康度
2. 竞争优势
3. 估值水平
4. 成长潜力

给出明确的估值判断。""",
            tools=[built_in_code_execution],  # 可以执行代码进行计算
        )
        
        # 策略 Agent
        self.strategy_agent = Agent(
            model="gemini-2.0-flash-exp",
            name="strategist",
            description="投资策略师",
            instruction="""你给出最终的投资建议。
考虑因素:
1. 分析结论
2. 市场环境
3. 风险收益比
4. 持仓建议

输出格式:
- 评级: 强烈买入/买入/持有/减持/卖出
- 目标价
- 止损位
- 核心逻辑
- 风险因素""",
        )
        
        # 根 Agent - 协调团队
        self.root_agent = Agent(
            model="gemini-2.0-flash-exp",
            name="investment_team",
            description="投研团队协调员",
            instruction="""你是投研团队负责人，协调数据收集、分析和策略三个环节。

工作流程:
1. 让 data_collector 收集数据
2. 让 analyst 进行分析
3. 让 strategist 给出建议
4. 整合输出最终报告

确保每个环节的质量，不通过就要求重做。""",
            sub_agents=[self.data_agent, self.analysis_agent, self.strategy_agent],
        )
        
        self.runner = Runner(agent=self.root_agent)
    
    async def research(self, stock_name: str) -> str:
        """运行投研流程"""
        session = Session()
        
        result = await self.runner.run_async(
            session=session,
            query=f"请对 {stock_name} 进行完整的投研分析",
        )
        
        return result.response


# ============ 代码执行 Agent ============

class CodeAnalysisAgent:
    """代码分析 Agent - 可以执行 Python 代码"""
    
    def __init__(self):
        self.agent = Agent(
            model="gemini-2.0-flash-exp",
            name="code_analyst",
            description="代码分析助手",
            instruction="""你是数据分析专家，可以编写和执行 Python 代码。

你可以:
1. 使用 built_in_code_execution 执行 Python 代码
2. 进行财务计算和建模
3. 绘制图表
4. 统计分析

每次回复都要包含代码块，代码会被自动执行。""",
            tools=[built_in_code_execution, google_search],
        )
        self.runner = Runner(agent=self.agent)
    
    async def analyze(self, task: str) -> str:
        """执行分析任务"""
        session = Session()
        result = await self.runner.run_async(session=session, query=task)
        return result.response


# ============ 演示 ============

async def demo_single_agent():
    """演示单 Agent"""
    print("=" * 60)
    print("Google ADK - 单 Agent 演示")
    print("=" * 60)
    
    if not GOOGLE_API_KEY:
        print("⚠️ 需要 GOOGLE_API_KEY")
        return
    
    agent = StockAnalysisAgent()
    
    print("\n🎯 分析: 贵州茅台")
    result = await agent.analyze("贵州茅台")
    print(f"\n📊 结果:\n{result}")


async def demo_multi_agent():
    """演示多 Agent 协作"""
    print("\n" + "=" * 60)
    print("Google ADK - 多 Agent 协作演示")
    print("=" * 60)
    
    if not GOOGLE_API_KEY:
        print("⚠️ 需要 GOOGLE_API_KEY")
        return
    
    team = InvestmentTeam()
    
    print("\n🎯 团队分析: 比亚迪")
    result = await team.research("比亚迪")
    print(f"\n📊 结果:\n{result}")


async def demo_code_execution():
    """演示代码执行"""
    print("\n" + "=" * 60)
    print("Google ADK - 代码执行演示")
    print("=" * 60)
    
    if not GOOGLE_API_KEY:
        print("⚠️ 需要 GOOGLE_API_KEY")
        return
    
    agent = CodeAnalysisAgent()
    
    task = """
    请计算以下股票的 ROE 排名，并绘制柱状图：
    贵州茅台: ROE=25.3%
    五粮液: ROE=20.1%
    比亚迪: ROE=15.6%
    宁德时代: ROE=18.9%
    美的集团: ROE=22.4%
    """
    
    print(f"\n📝 任务: {task[:80]}...")
    result = await agent.analyze(task)
    print(f"\n✅ 结果:\n{result}")


async def main():
    """主函数"""
    # await demo_single_agent()
    # await demo_multi_agent()
    await demo_code_execution()


if __name__ == "__main__":
    asyncio.run(main())
