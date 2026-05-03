#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Agent协作 - Manager-Worker 模式

面试考点: Q5(多Agent系统), Q18(投研平台四类Agent协作)
核心思想: 管理者分配任务，工作者并行执行
"""

import json
import time
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class Task:
    """任务定义"""
    id: str
    description: str
    agent_type: str
    input_data: Dict
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending / running / completed / failed
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class Agent:
    """Agent定义"""
    name: str
    agent_type: str
    description: str
    llm_call: Callable[[str], str]
    system_prompt: str = ""


class ManagerWorkerSystem:
    """
    Manager-Worker 多Agent协作系统
    
    面试要点:
    - 管理者负责任务分解和结果汇总
    - 工作者各自执行子任务
    - 通过共享上下文传递状态
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.shared_context: Dict = {}
        self.execution_log: List[Dict] = []
    
    def register_agent(self, agent: Agent):
        """注册Agent"""
        self.agents[agent.name] = agent
    
    def create_task(self, task_id: str, description: str, 
                   agent_type: str, input_data: Dict,
                   dependencies: List[str] = None) -> Task:
        """创建任务"""
        task = Task(
            id=task_id,
            description=description,
            agent_type=agent_type,
            input_data=input_data,
            dependencies=dependencies or []
        )
        self.tasks[task_id] = task
        return task
    
    def _execute_single_task(self, task: Task) -> Task:
        """执行单个任务"""
        # 查找合适的Agent
        agent = None
        for a in self.agents.values():
            if a.agent_type == task.agent_type:
                agent = a
                break
        
        if not agent:
            task.status = "failed"
            task.error = f"找不到类型为'{task.agent_type}'的Agent"
            return task
        
        # 检查依赖
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != "completed":
                    task.status = "failed"
                    task.error = f"依赖任务'{dep_id}'未完成"
                    return task
                # 将依赖结果注入输入
                task.input_data[f"dep_{dep_id}"] = dep_task.result
        
        # 执行任务
        task.status = "running"
        start_time = time.time()
        
        try:
            # 构建Prompt
            context_str = json.dumps(self.shared_context, ensure_ascii=False)
            input_str = json.dumps(task.input_data, ensure_ascii=False)
            
            prompt = f"""{agent.system_prompt}

共享上下文:
{context_str}

任务: {task.description}
输入数据: {input_str}

请完成任务并输出结果。"""
            
            result = agent.llm_call(prompt)
            task.result = result
            task.status = "completed"
            
            # 更新共享上下文
            self.shared_context[task.id] = {
                "description": task.description,
                "result": result[:200]  # 只保留摘要
            }
            
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
        
        task.execution_time_ms = (time.time() - start_time) * 1000
        
        self.execution_log.append({
            "task_id": task.id,
            "agent": agent.name,
            "status": task.status,
            "time_ms": task.execution_time_ms
        })
        
        return task
    
    def execute_dag(self, tasks: List[Task]) -> Dict[str, Task]:
        """
        执行DAG任务流
        
        面试要点: 按DAG顺序执行，支持并行
        """
        print(f"\n🎯 开始执行DAG，共{len(tasks)}个任务")
        print("=" * 60)
        
        # 拓扑排序
        executed = set()
        pending = set(t.id for t in tasks)
        
        while pending:
            # 找到可以执行的任务（依赖已满足）
            ready = []
            for task_id in pending:
                task = self.tasks[task_id]
                if all(dep in executed for dep in task.dependencies):
                    ready.append(task)
            
            if not ready:
                # 死锁检测
                print("❌ 死锁 detected! 无法继续执行")
                break
            
            print(f"\n📦 本轮可执行任务: {[t.id for t in ready]}")
            
            # 并行执行
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._execute_single_task, task): task 
                          for task in ready}
                
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        status_emoji = "✅" if result.status == "completed" else "❌"
                        print(f"  {status_emoji} {result.id}: {result.status} ({result.execution_time_ms:.0f}ms)")
                    except Exception as e:
                        print(f"  ❌ {task.id}: 异常 - {str(e)}")
            
            # 更新状态
            for task in ready:
                executed.add(task.id)
                pending.remove(task.id)
        
        return self.tasks
    
    def generate_report(self) -> Dict:
        """生成执行报告"""
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        total_time = sum(t.execution_time_ms for t in self.tasks.values())
        
        return {
            "total_tasks": len(self.tasks),
            "completed": completed,
            "failed": failed,
            "success_rate": completed / len(self.tasks) if self.tasks else 0,
            "total_time_ms": total_time,
            "avg_time_ms": total_time / len(self.tasks) if self.tasks else 0
        }


# ============ 模拟LLM ============

class MockLLM:
    """模拟不同Agent的LLM"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
    
    def __call__(self, prompt: str) -> str:
        """根据Agent类型返回不同结果"""
        if self.agent_type == "financial_analyst":
            return self._financial_analysis(prompt)
        elif self.agent_type == "sentiment_analyst":
            return self._sentiment_analysis(prompt)
        elif self.agent_type == "rotation_analyst":
            return self._rotation_analysis(prompt)
        elif self.agent_type == "quant_analyst":
            return self._quant_analysis(prompt)
        elif self.agent_type == "report_writer":
            return self._write_report(prompt)
        return "分析完成"
    
    def _financial_analysis(self, prompt: str) -> str:
        return """财务分析结果:
- ROE: 18.5% (优秀)
- ROIC: 15.2% (良好)
- 自由现金流: 正
- 基本面评分: 85/100"""
    
    def _sentiment_analysis(self, prompt: str) -> str:
        return """情绪分析结果:
- 涨停数: 45家
- 龙头状态: 强势
- 资金流向: 净流入
- 情绪评分: 75/100"""
    
    def _rotation_analysis(self, prompt: str) -> str:
        return """轮动分析结果:
- 当前主线: 半导体
- 传导路径: 半导体→设备→材料
- 轮动概率: 68%
- 建议: 关注设备端"""
    
    def _quant_analysis(self, prompt: str) -> str:
        return """回测结果:
- 策略: 双均线
- 年化收益: 12.3%
- 最大回撤: -8.5%
- 夏普比率: 1.45"""
    
    def _write_report(self, prompt: str) -> str:
        return """投研报告:
综合评分: 78/100
建议: 谨慎乐观
核心逻辑: 基本面优秀+情绪积极+轮动确认"""


def demo():
    """演示多Agent协作"""
    
    system = ManagerWorkerSystem(max_workers=4)
    
    # 注册Agent
    agents_config = [
        ("财报分析师", "financial_analyst", "分析公司财务数据，计算ROE/ROIC等指标"),
        ("情绪分析师", "sentiment_analyst", "分析板块情绪和资金流向"),
        ("轮动分析师", "rotation_analyst", "分析板块轮动概率和传导链"),
        ("量化分析师", "quant_analyst", "对策略进行历史回测"),
        ("报告撰写员", "report_writer", "整合各Agent结果，生成最终报告"),
    ]
    
    for name, agent_type, desc in agents_config:
        system.register_agent(Agent(
            name=name,
            agent_type=agent_type,
            description=desc,
            llm_call=MockLLM(agent_type),
            system_prompt=f"你是{name}，{desc}"
        ))
    
    print("=" * 60)
    print("多Agent协作系统 - Manager-Worker模式")
    print("=" * 60)
    
    # 创建DAG任务
    # 财报分析 和 情绪分析 可以并行
    # 轮动分析 依赖 情绪分析
    # 量化回测 依赖 财报分析
    # 报告撰写 依赖 所有分析
    
    tasks = [
        system.create_task("T1", "分析沪电股份财务数据", "financial_analyst", 
                          {"stock": "沪电股份"}),
        system.create_task("T2", "分析半导体板块情绪", "sentiment_analyst", 
                          {"sector": "半导体"}),
        system.create_task("T3", "分析板块轮动", "rotation_analyst", 
                          {"sector": "半导体"}, 
                          dependencies=["T2"]),
        system.create_task("T4", "回测双均线策略", "quant_analyst", 
                          {"strategy": "双均线", "period": "2024-2025"},
                          dependencies=["T1"]),
        system.create_task("T5", "生成投研报告", "report_writer", 
                          {"target": "沪电股份"},
                          dependencies=["T1", "T2", "T3", "T4"]),
    ]
    
    # 执行DAG
    results = system.execute_dag(tasks)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📊 各Agent分析结果:")
    for task_id, task in results.items():
        status = "✅" if task.status == "completed" else "❌"
        print(f"\n{status} {task_id} ({task.agent_type}):")
        if task.result:
            print(f"   {task.result[:150]}...")
        if task.error:
            print(f"   错误: {task.error}")
    
    # 执行报告
    print("\n" + "=" * 60)
    report = system.generate_report()
    print("📈 执行统计:")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo()
