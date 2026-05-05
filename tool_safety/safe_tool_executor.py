#!/usr/bin/env python3
"""
工具安全执行器 - LangChain BaseTool包装
面试考点: Q2(工具安全), Q7(Agent安全性)
"""
import json, time, hashlib
from typing import Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.callbacks import CallbackManagerForToolRun


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolCall:
    tool_name: str
    params: Dict
    risk: RiskLevel
    timestamp: float = field(default_factory=time.time)
    result: Optional[str] = None
    error: Optional[str] = None
    approved: bool = False
    trace_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])


class AuditLog:
    """审计日志"""
    def __init__(self):
        self.logs: List[ToolCall] = []

    def log(self, call: ToolCall):
        self.logs.append(call)
        # 异常检测
        recent = [l for l in self.logs if l.tool_name == call.tool_name and call.timestamp - l.timestamp < 600]
        if len(recent) > 10:
            print(f"  🚨 异常: '{call.tool_name}' 10分钟调用{len(recent)}次")

    def report(self) -> Dict:
        total = len(self.logs)
        return {
            "total": total,
            "errors": sum(1 for l in self.logs if l.error),
            "critical": sum(1 for l in self.logs if l.risk == RiskLevel.CRITICAL),
            "unique_tools": len(set(l.tool_name for l in self.logs))
        }


class SafeTool(BaseTool):
    """安全工具基类 (继承BaseTool)"""
    risk_level: RiskLevel = RiskLevel.LOW
    needs_approval: bool = False
    audit: Optional[AuditLog] = None
    approval_fn: Optional[Callable] = None

    def _run(self, *args, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs):
        start = time.time()
        call = ToolCall(tool_name=self.name, params=kwargs, risk=self.risk_level)

        if self.needs_approval:
            if self.approval_fn:
                if not self.approval_fn(self.name, kwargs):
                    call.error = "审批被拒绝"
                    self.audit and self.audit.log(call)
                    raise ToolException("审批被拒绝")
                call.approved = True
            elif self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                call.error = "高风险操作需配置审批"
                self.audit and self.audit.log(call)
                raise ToolException("未配置审批回调")

        try:
            result = self._safe_run(*args, run_manager=run_manager, **kwargs)
            call.result = str(result)
            call.execution_time_ms = (time.time() - start) * 1000
            self.audit and self.audit.log(call)
            return result
        except Exception as e:
            call.error = str(e)
            call.execution_time_ms = (time.time() - start) * 1000
            self.audit and self.audit.log(call)
            raise ToolException(f"执行失败: {e}")

    def _safe_run(self, *args, run_manager=None, **kwargs):
        raise NotImplementedError

    async def _arun(self, *args, run_manager=None, **kwargs):
        return self._run(*args, run_manager=run_manager, **kwargs)


class ToolRegistry:
    """工具注册表"""
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.audit = AuditLog()
        self.approvers: Dict[RiskLevel, Callable] = {}

    def set_approver(self, level: RiskLevel, fn: Callable):
        self.approvers[level] = fn

    def register(self, tool: BaseTool, risk: RiskLevel = RiskLevel.LOW, needs_approval: bool = False):
        if isinstance(tool, SafeTool):
            tool.audit = self.audit
            tool.risk_level = risk
            tool.needs_approval = needs_approval
            tool.approval_fn = self.approvers.get(risk)
        self.tools[tool.name] = tool
        print(f"  ✅ {tool.name} ({risk.value})")

    def create(self, name: str, fn: Callable, schema: Type[BaseModel], desc: str,
               risk: RiskLevel = RiskLevel.LOW, needs_approval: bool = False) -> StructuredTool:
        tool = StructuredTool.from_function(func=fn, name=name, description=desc, args_schema=schema)
        # 包装为SafeTool逻辑
        self.register(tool, risk, needs_approval)
        return tool

    def execute(self, name: str, params: Dict, dry_run: bool = False) -> Dict:
        if name not in self.tools:
            return {"ok": False, "error": f"未知工具: {name}"}
        print(f"\n🔧 {name}({json.dumps(params, ensure_ascii=False)})")
        if dry_run:
            return {"ok": True, "result": "[Dry-run]", "dry_run": True}
        try:
            return {"ok": True, "result": self.tools[name].invoke(params)}
        except ToolException as e:
            return {"ok": False, "error": str(e)}

    def all_tools(self) -> List[BaseTool]:
        return list(self.tools.values())


# ============ 示例工具 ============

class QueryInput(BaseModel):
    code: str = Field(description="股票代码")

class UpdateInput(BaseModel):
    code: str = Field(description="股票代码")
    shares: int = Field(description="股数", ge=0)

class DeleteInput(BaseModel):
    order_id: str = Field(description="订单ID")

class TransferInput(BaseModel):
    to: str = Field(description="目标账户")
    amount: float = Field(description="金额", gt=0)


def query_stock(code: str) -> str:
    return f"{code}: 45.2元, +2.3%"

def update_portfolio(code: str, shares: int) -> str:
    return f"更新: {code} {shares}股"

def delete_order(order_id: str) -> str:
    return f"订单{order_id}已删除"

def transfer_funds(to: str, amount: float) -> str:
    return f"转账{amount}元到{to}"


def demo():
    print("=" * 50)
    print("Tool Safety - LangChain")
    print("=" * 50)

    reg = ToolRegistry()
    reg.set_approver(RiskLevel.HIGH, lambda n, p: (print(f"  ⏸️ 拒绝 {n}"), False)[1])
    reg.set_approver(RiskLevel.CRITICAL, lambda n, p: (print(f"  ⏸️ 拒绝 {n}"), False)[1])

    reg.create("query", query_stock, QueryInput, "查询股票", RiskLevel.LOW)
    reg.create("update", update_portfolio, UpdateInput, "更新持仓", RiskLevel.MEDIUM)
    reg.create("delete", delete_order, DeleteInput, "删除订单", RiskLevel.HIGH, True)
    reg.create("transfer", transfer_funds, TransferInput, "转账", RiskLevel.CRITICAL, True)

    # 场景
    print("\n📋 正常查询")
    print(reg.execute("query", {"code": "002463"}))

    print("\n📋 参数错误(Pydantic拦截)")
    print(reg.execute("query", {"code": 12345}))

    print("\n📋 更新持仓")
    print(reg.execute("update", {"code": "002463", "shares": 100}))

    print("\n📋 删除订单(需审批)")
    print(reg.execute("delete", {"order_id": "ORD-001"}))

    print("\n📋 转账(Dry-run)")
    print(reg.execute("transfer", {"to": "622202", "amount": 10000}, dry_run=True))

    print("\n📋 范围校验")
    print(reg.execute("update", {"code": "002463", "shares": -100}))

    print("\n📊 审计报告:")
    print(reg.audit.report())


if __name__ == "__main__":
    demo()
