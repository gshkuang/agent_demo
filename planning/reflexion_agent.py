#!/usr/bin/env python3
"""
Reflexion Agent - LangGraph版本
面试考点: Q2(自我反思), Q6(Tree of Thoughts)
"""
from typing import TypedDict, List, Optional, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class ReflexionState(TypedDict):
    task: str
    messages: List
    attempt: int
    max_attempts: int
    result: Optional[str]
    success: bool
    memory: List[str]


def build_reflexion_agent(llm, max_attempts: int = 3):
    """构建Reflexion Agent (LangGraph)"""
    sys_prompt = "你是带反思能力的Agent。执行任务→评估→改进→重试。"

    def should_reflect(state: ReflexionState):
        if state["success"] or state["attempt"] >= state["max_attempts"]:
            return "end"
        return "reflect"

    def execute(state: ReflexionState):
        memory = "\n".join(state["memory"][-3:]) if state["memory"] else "暂无"
        prompt = f"经验教训:\n{memory}\n\n任务: {state['task']}"
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=prompt), response],
            "attempt": state["attempt"] + 1,
            "result": response.content,
            "success": False
        }

    def reflect(state: ReflexionState):
        prompt = f"任务: {state['task']}\n输出: {state['result']}\n\nCritique: <批评>\nImprovement: <改进>\nSuccess: <true/false>"
        r = llm.invoke([HumanMessage(content=prompt)])
        text = r.content
        success = "success: true" in text.lower() or "success:true" in text.lower()
        memory = state["memory"] + [f"{'成功' if success else '失败'}: {text[:100]}"]
        return {**state, "success": success, "memory": memory}

    workflow = StateGraph(ReflexionState)
    workflow.add_node("execute", execute)
    workflow.add_node("reflect", reflect)
    workflow.set_entry_point("execute")
    workflow.add_conditional_edges("execute", should_reflect, {"reflect": "reflect", "end": END})
    workflow.add_conditional_edges("reflect", lambda s: "execute" if not s["success"] and s["attempt"] < s["max_attempts"] else "end")
    return workflow.compile()


def demo():
    print("=" * 50)
    print("Reflexion Agent - LangGraph")
    print("=" * 50)

    try:
        import os
        llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"], temperature=0)
        agent = build_reflexion_agent(llm, max_attempts=3)

        result = agent.invoke({
            "task": "计算沪电股份的ROE",
            "messages": [], "attempt": 0, "max_attempts": 3,
            "result": None, "success": False, "memory": []
        })

        print(f"\n尝试: {result['attempt']}, 成功: {result['success']}")
        print(f"结果: {result.get('result', 'N/A')[:100]}")
        print(f"经验: {len(result['memory'])}条")

    except Exception as e:
        print(f"\n⚠️ 需要OPENAI_API_KEY: {e}")


if __name__ == "__main__":
    demo()
