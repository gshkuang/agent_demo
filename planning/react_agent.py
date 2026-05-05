#!/usr/bin/env python3
"""
ReAct Agent - LangGraph版本
面试考点: Q1(ReAct vs Plan-and-Execute), Q8(ReAct框架详解)
"""
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: List
    current_step: int
    max_steps: int


@tool
def search_stock(query: str) -> str:
    """搜索股票基本信息"""
    stocks = {"沪电股份": "代码002463，PCB龙头，2024年ROE 18.5%"}
    return next((v for k, v in stocks.items() if k in query), f"未找到'{query}'")


@tool
def calculate_roe(net_profit: float, equity: float) -> str:
    """计算ROE"""
    return f"ROE = {(net_profit / equity * 100):.2f}%" if equity > 0 else "净资产必须大于0"


@tool
def get_market_sentiment() -> str:
    """获取市场情绪"""
    return "今日涨停45家，跌停12家，情绪偏乐观"


def build_react_agent(llm, tools: list, system_prompt: str = None):
    """构建ReAct Agent (LangGraph)"""
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = system_prompt or "你是ReAct Agent，通过推理和行动交替完成任务。"
    tool_node = ToolNode(tools)

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["current_step"] < state["max_steps"]:
            return "continue"
        return "end"

    def call_model(state: AgentState):
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(msgs)
        return {
            "messages": state["messages"] + [response],
            "current_step": state["current_step"] + 1,
            "max_steps": state["max_steps"]
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
    workflow.add_edge("action", "agent")
    return workflow.compile()


def demo():
    print("=" * 50)
    print("ReAct Agent - LangGraph")
    print("=" * 50)

    try:
        import os
        llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"], temperature=0)
        agent = build_react_agent(llm, [search_stock, calculate_roe, get_market_sentiment])

        result = agent.invoke({
            "messages": [HumanMessage(content="分析沪电股份投资价值")],
            "current_step": 0,
            "max_steps": 5
        })

        print(f"\n步数: {result['current_step']}")
        for i, msg in enumerate(result["messages"]):
            role = type(msg).__name__.replace("Message", "")
            content = getattr(msg, "content", str(msg))[:80]
            tool = f" [tools: {[t['name'] for t in msg.tool_calls]}]" if getattr(msg, "tool_calls", None) else ""
            print(f"  {i+1}. {role}: {content}...{tool}")

    except Exception as e:
        print(f"\n⚠️ 需要OPENAI_API_KEY: {e}")


if __name__ == "__main__":
    demo()
