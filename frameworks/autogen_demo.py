#!/usr/bin/env python3
"""
AutoGen 多 Agent 协作演示
P2: Microsoft AutoGen 框架 - 对话式多 Agent 系统

场景: 投研报告生成
- Researcher: 收集数据
- Analyst: 分析数据
- Writer: 撰写报告
- Reviewer: 审核报告
"""
import os
import asyncio
from typing import Dict, List, Optional

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    from autogen.coding import LocalCommandLineCodeExecutor
except ImportError:
    print("请先安装 AutoGen: pip install pyautogen")
    raise


# ============ 配置 ============

LLM_CONFIG = {
    "config_list": [{
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
    }],
    "temperature": 0.7,
    "timeout": 120,
}


# ============ Agent 定义 ============

class ResearchTeam:
    """投研团队 - AutoGen 多 Agent 协作"""
    
    def __init__(self):
        self.agents = {}
        self._create_agents()
    
    def _create_agents(self):
        """创建团队成员"""
        
        # 1. 研究员 - 收集数据
        self.agents["researcher"] = ConversableAgent(
            name="Researcher",
            system_message="""你是资深行业研究员，擅长收集和整理公司基本面数据。
你的任务是：
1. 收集目标公司的财务数据（营收、利润、ROE、PE、PB）
2. 收集行业竞争格局信息
3. 收集最新新闻和公告
4. 输出结构化的数据摘要

你只负责收集数据，不做投资建议。""",
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
        )
        
        # 2. 分析师 - 分析数据
        self.agents["analyst"] = ConversableAgent(
            name="Analyst",
            system_message="""你是资深投资分析师，擅长财务分析和估值建模。
你的任务是：
1. 分析公司的财务健康状况
2. 进行同行对比分析
3. 评估公司的竞争优势和护城河
4. 给出估值判断（低估/合理/高估）

基于研究员提供的数据进行分析，输出分析结论。""",
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
        )
        
        # 3. 撰稿人 - 撰写报告
        self.agents["writer"] = ConversableAgent(
            name="Writer",
            system_message="""你是专业投研报告撰稿人。
你的任务是：
1. 将研究员和分析师的输出整合为结构化的投研报告
2. 报告包含：公司概况、财务分析、行业地位、投资建议
3. 语言专业、逻辑清晰
4. 输出 Markdown 格式

你只负责撰写，不添加新的数据或观点。""",
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
        )
        
        # 4. 审核员 - 质量把控
        self.agents["reviewer"] = ConversableAgent(
            name="Reviewer",
            system_message="""你是投研报告质量审核员。
你的任务是：
1. 检查报告是否有数据错误
2. 检查逻辑是否自洽
3. 检查投资建议是否有充分依据
4. 如果发现问题，要求修改；如果通过，回复 "APPROVE"

严格把关，不通过就指出具体问题。""",
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER",
        )
        
        # 5. 用户代理 - 人类入口
        self.agents["user"] = ConversableAgent(
            name="User",
            system_message="你是用户，提出投研需求。",
            llm_config=False,
            human_input_mode="ALWAYS",
        )
    
    def run_research(self, stock_name: str, max_rounds: int = 10) -> str:
        """
        运行投研流程
        
        流程: User -> Researcher -> Analyst -> Writer -> Reviewer
        如果 Reviewer 不通过，返回 Writer 修改
        """
        
        # 创建群聊
        groupchat = GroupChat(
            agents=[
                self.agents["researcher"],
                self.agents["analyst"],
                self.agents["writer"],
                self.agents["reviewer"],
            ],
            messages=[],
            max_round=max_rounds,
            speaker_selection_method="round_robin",  # 轮流发言
        )
        
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=LLM_CONFIG,
        )
        
        # 启动任务
        task = f"请对 {stock_name} 进行深度投研分析，输出一份完整的投研报告。"
        
        result = self.agents["user"].initiate_chat(
            manager,
            message=task,
            clear_history=True,
        )
        
        return result.summary


# ============ 代码执行 Agent ============

class CodeExecutionAgent:
    """代码执行 Agent - 可以写代码、运行代码、分析结果"""
    
    def __init__(self):
        # 创建代码执行器（沙箱环境）
        self.executor = LocalCommandLineCodeExecutor(
            timeout=60,
            work_dir="./coding_workspace",
        )
        
        # 代码助手
        self.coder = ConversableAgent(
            name="Coder",
            system_message="""你是 Python 数据分析师。你可以：
1. 编写 Python 代码进行数据分析
2. 使用 pandas、numpy、matplotlib 等库
3. 运行代码并解释结果

每次回复都要包含代码块，代码会被自动执行。""",
            llm_config=LLM_CONFIG,
            code_execution_config={"executor": self.executor},
            human_input_mode="NEVER",
        )
        
        # 用户代理
        self.user = ConversableAgent(
            name="User",
            system_message="提出数据分析需求",
            llm_config=False,
            human_input_mode="NEVER",
        )
    
    def analyze(self, task: str) -> str:
        """执行数据分析任务"""
        result = self.user.initiate_chat(
            self.coder,
            message=task,
            clear_history=True,
        )
        return result.summary


# ============ 演示 ============

def demo_research_team():
    """演示投研团队协作"""
    print("=" * 60)
    print("AutoGen 投研团队协作演示")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 需要 OPENAI_API_KEY")
        return
    
    team = ResearchTeam()
    
    print("\n🎯 任务: 分析贵州茅台")
    print("-" * 40)
    
    result = team.run_research("贵州茅台", max_rounds=8)
    
    print("\n📊 最终报告:")
    print(result)


def demo_code_execution():
    """演示代码执行 Agent"""
    print("\n" + "=" * 60)
    print("AutoGen 代码执行 Agent 演示")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 需要 OPENAI_API_KEY")
        return
    
    agent = CodeExecutionAgent()
    
    task = """
    请分析以下股票数据，计算每只股票的 ROE 排名，并绘制柱状图：
    
    股票数据:
    - 贵州茅台: ROE=25.3%, PE=28.5, PB=8.2
    - 五粮液: ROE=20.1%, PE=18.2, PB=4.5
    - 比亚迪: ROE=15.6%, PE=32.1, PB=5.8
    - 宁德时代: ROE=18.9%, PE=25.8, PB=6.2
    - 美的集团: ROE=22.4%, PE=14.2, PB=3.1
    """
    
    print(f"\n📝 任务: {task[:100]}...")
    result = agent.analyze(task)
    print(f"\n✅ 结果:\n{result}")


if __name__ == "__main__":
    demo_research_team()
    # demo_code_execution()
