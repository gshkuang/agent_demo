#!/usr/bin/env python3
"""
CrewAI 多 Agent 团队协作演示
P2: CrewAI 框架 - 角色驱动的 Agent 团队

场景: 投研报告生成（与 AutoGen 对比）
CrewAI 特点:
- 基于角色的任务分配
- 明确的任务依赖关系
- 支持工具集成
- 过程可观测
"""
import os
from typing import List, Dict
from datetime import datetime

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
except ImportError:
    print("请先安装 CrewAI: pip install crewai")
    raise


# ============ 工具定义 ============

class StockSearchTool(BaseTool):
    """股票搜索工具"""
    name: str = "search_stock"
    description: str = "搜索股票的基本信息，输入股票代码或名称"
    
    def _run(self, query: str) -> str:
        stocks = {
            "600519": "贵州茅台 - 白酒龙头，ROE 25.3%，PE 28.5x",
            "000858": "五粮液 - 白酒老二，ROE 20.1%，PE 18.2x",
            "002594": "比亚迪 - 新能源整车，ROE 15.6%，PE 32.1x",
        }
        return stocks.get(query, f"未找到 {query}，请尝试其他代码")


class MarketDataTool(BaseTool):
    """市场数据工具"""
    name: str = "market_data"
    description: str = "获取市场情绪和板块数据"
    
    def _run(self, sector: str = "all") -> str:
        data = {
            "半导体": "涨幅+2.3%，资金净流入+15亿",
            "新能源": "涨幅+1.8%，资金净流入+8亿",
            "白酒": "涨幅-0.5%，资金净流出-3亿",
        }
        if sector == "all":
            return "\n".join([f"{k}: {v}" for k, v in data.items()])
        return data.get(sector, "暂无数据")


# ============ Agent 团队 ============

class InvestmentResearchCrew:
    """投研团队 - CrewAI 实现"""
    
    def __init__(self):
        self.tools = [StockSearchTool(), MarketDataTool()]
        self._create_agents()
    
    def _create_agents(self):
        """创建团队成员"""
        
        # 1. 数据收集员
        self.researcher = Agent(
            role="资深行业研究员",
            goal="收集目标公司的全面数据，包括财务、行业、竞争格局",
            backstory="""你有10年行业研究经验，曾在顶级券商研究所工作。
你擅长从各种渠道收集数据，包括财报、公告、新闻、行业报告。
你的数据收集能力是整个团队的基础。""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
        )
        
        # 2. 财务分析师
        self.analyst = Agent(
            role="CFA 持证分析师",
            goal="基于收集的数据进行深度财务分析和估值",
            backstory="""你持有 CFA 证书，专注于基本面分析和估值建模。
你擅长杜邦分析、DCF 估值、同行对比等方法。
你的分析结论是投资建议的核心依据。""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
        )
        
        # 3. 策略师
        self.strategist = Agent(
            role="首席策略师",
            goal="结合宏观和市场情绪，给出投资策略建议",
            backstory="""你有15年投资经验，管理过百亿资金。
你擅长宏观判断、择时、风险控制。
你的建议直接影响投资组合的构建。""",
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
        )
        
        # 4. 报告撰写员
        self.writer = Agent(
            role="投研报告撰写专家",
            goal="将分析结果整合为专业的投研报告",
            backstory="""你是顶级投研报告撰写人，曾获得新财富最佳分析师。
你的报告逻辑清晰、数据准确、观点鲜明。
你的报告直接影响投资决策。""",
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
        )
    
    def create_tasks(self, stock_code: str, stock_name: str) -> List[Task]:
        """创建任务序列"""
        
        task1 = Task(
            description=f"""
            收集 {stock_name}({stock_code}) 的全面数据：
            1. 使用 search_stock 工具获取基本信息
            2. 使用 market_data 工具了解所属板块情况
            3. 整理成结构化的数据摘要
            
            输出格式:
            - 公司概况
            - 财务数据摘要
            - 行业地位
            - 最新动态
            """,
            expected_output="结构化的数据收集报告",
            agent=self.researcher,
        )
        
        task2 = Task(
            description=f"""
            基于研究员收集的数据，对 {stock_name} 进行深度分析：
            1. 财务健康度评估（ROE、现金流、负债率）
            2. 同行对比分析
            3. 竞争优势和护城河评估
            4. 估值判断（PE/PB 分位数）
            
            输出格式:
            - 财务分析
            - 竞争分析
            - 估值判断
            """,
            expected_output="深度分析报告",
            agent=self.analyst,
            context=[task1],  # 依赖 task1 的输出
        )
        
        task3 = Task(
            description=f"""
            结合市场情绪和宏观环境，给出 {stock_name} 的投资策略：
            1. 当前市场环境评估
            2. 板块轮动判断
            3. 买入/持有/卖出建议
            4. 目标价位和止损位
            
            输出格式:
            - 市场环境
            - 策略建议
            - 风险提示
            """,
            expected_output="投资策略建议",
            agent=self.strategist,
            context=[task1, task2],
        )
        
        task4 = Task(
            description=f"""
            整合所有分析，撰写完整的投研报告：
            
            报告结构:
            # {stock_name}({stock_code}) 深度研究报告
            
            ## 一、投资摘要
            （核心观点和建议）
            
            ## 二、公司概况
            （研究员数据）
            
            ## 三、财务分析
            （分析师结论）
            
            ## 四、竞争格局
            （行业地位）
            
            ## 五、投资策略
            （策略师建议）
            
            ## 六、风险提示
            
            ## 七、盈利预测
            
            要求:
            - 专业、客观、数据驱动
            - 逻辑清晰，层次分明
            - 字数不少于2000字
            """,
            expected_output="完整的 Markdown 格式投研报告",
            agent=self.writer,
            context=[task1, task2, task3],
        )
        
        return [task1, task2, task3, task4]
    
    def run(self, stock_code: str, stock_name: str) -> str:
        """运行投研流程"""
        tasks = self.create_tasks(stock_code, stock_name)
        
        crew = Crew(
            agents=[self.researcher, self.analyst, self.strategist, self.writer],
            tasks=tasks,
            process=Process.sequential,  # 顺序执行
            verbose=True,
            memory=True,  # 启用记忆
        )
        
        result = crew.kickoff()
        return result


# ============ 演示 ============

def demo():
    """演示 CrewAI 投研团队"""
    print("=" * 60)
    print("CrewAI 投研团队协作演示")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 需要 OPENAI_API_KEY")
        return
    
    crew = InvestmentResearchCrew()
    
    print("\n🎯 任务: 分析贵州茅台(600519)")
    print("-" * 40)
    
    result = crew.run("600519", "贵州茅台")
    
    print("\n📊 最终报告:")
    print(result)


if __name__ == "__main__":
    demo()
