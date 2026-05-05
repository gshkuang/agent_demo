#!/usr/bin/env python3
"""
多Agent协作 - Manager-Worker (LangGraph版本)
面试考点: Q5(多Agent系统), Q18(投研平台四类Agent协作)
"""
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json


class ManagerState(TypedDict):
    original_task: str
    workers: List[Dict]
    completed_results: Dict[str, str]
    final_report: Optional[str]


def build_manager_worker_graph(llm, agent_configs: List[Dict]):
    """构建Manager-Worker图 (LangGraph Send动态分发)"""
    workers_map = {cfg["type"]: cfg for cfg in agent_configs}

    def manager_decompose(state: ManagerState):
        agents_info = "\n".join(f"- {c['name']} ({c['type']}): {c['description']}" for c in agent_configs)
        prompt = f"分解任务为JSON: {{'tasks':[{{'task_id':'T1','description':'...','agent_type':'...','dependencies':[]}}]}}\n\nAgent:\n{agents_info}\n\n任务: {state['original_task']}"
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            text = response.content
            if "```" in text: text = text.split("```")[1].split("```")[0]
            tasks = json.loads(text.strip()).get("tasks", [])
            workers = [{"task_id": t["task_id"], "task_description": t["description"],
                       "agent_type": t["agent_type"], "input_data": t.get("input_data", {}),
                       "dependencies_results": {}, "result": None, "status": "pending"} for t in tasks]
            return {"workers": workers}
        except:
            return {"workers": [{"task_id": "T1", "task_description": state["original_task"],
                                "agent_type": agent_configs[0]["type"], "input_data": {},
                                "dependencies_results": {}, "result": None, "status": "pending"}]}

    def dispatch(state: ManagerState):
        sends = []
        completed = state.get("completed_results", {})
        for w in state["workers"]:
            if w["status"] != "pending": continue
            deps = next((t.get("dependencies", []) for t in state.get("_plan", {}).get("tasks", []) if t["task_id"] == w["task_id"]), [])
            if all(d in completed for d in deps):
                w["dependencies_results"] = {d: completed[d] for d in deps}
                if w["agent_type"] in workers_map:
                    sends.append(Send(f"worker_{w['agent_type']}", w))
        return sends if sends else [Send("aggregator", state)]

    def create_worker(cfg: Dict):
        def worker_node(state: Dict):
            ctx = "\n".join(f"[{k}]: {v[:80]}" for k, v in state.get("dependencies_results", {}).items())
            prompt = f"{cfg.get('system_prompt', cfg['name'])}\n{ctx}\n任务: {state['task_description']}"
            try:
                r = llm.invoke([HumanMessage(content=prompt)])
                return {"result": r.content, "status": "completed"}
            except Exception as e:
                return {"result": None, "status": "failed", "error": str(e)}
        return worker_node

    def aggregator(state: ManagerState):
        summary = "\n\n".join(f"## {w['task_id']}\n{w.get('result', '无')[:150]}" for w in state["workers"])
        prompt = f"根据以下分析生成投研报告:\n{summary}\n\n任务: {state['original_task']}"
        r = llm.invoke([HumanMessage(content=prompt)])
        return {"final_report": r.content}

    workflow = StateGraph(ManagerState)
    workflow.add_node("manager", manager_decompose)
    for cfg in agent_configs:
        workflow.add_node(f"worker_{cfg['type']}", create_worker(cfg))
    workflow.add_node("aggregator", aggregator)
    workflow.set_entry_point("manager")
    workflow.add_conditional_edges("manager", dispatch, {f"worker_{c['type']}": f"worker_{c['type']}" for c in agent_configs})
    for cfg in agent_configs:
        workflow.add_edge(f"worker_{cfg['type']}", "aggregator")
    workflow.add_edge("aggregator", END)
    return workflow.compile()


def demo():
    print("=" * 50)
    print("Manager-Worker - LangGraph")
    print("=" * 50)

    try:
        import os
        llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"], temperature=0)
        configs = [
            {"name": "财报分析师", "type": "financial", "description": "分析财务数据", "system_prompt": "你是财报分析师。"},
            {"name": "情绪分析师", "type": "sentiment", "description": "分析市场情绪", "system_prompt": "你是情绪分析师。"},
            {"name": "报告撰写员", "type": "report", "description": "汇总报告", "system_prompt": "你是报告撰写员。"}
        ]
        graph = build_manager_worker_graph(llm, configs)
        result = graph.invoke({"original_task": "分析沪电股份", "workers": [], "completed_results": {}, "final_report": None})
        print(f"\n报告: {result.get('final_report', '无')[:300]}")

    except Exception as e:
        print(f"\n⚠️ 需要OPENAI_API_KEY: {e}")


if __name__ == "__main__":
    demo()
