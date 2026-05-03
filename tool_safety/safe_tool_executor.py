#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具安全执行器 - 三层防护

面试考点: Q2(工具安全), Q7(Agent安全性)
核心思想: Schema校验 + 业务兜底 + 审计追踪
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"       # 只读查询
    MEDIUM = "medium"  # 影响有限
    HIGH = "high"      # 重要操作
    CRITICAL = "critical"  # 资金/删除等


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    parameters: Dict
    risk_level: RiskLevel
    timestamp: float = field(default_factory=time.time)
    result: Optional[str] = None
    error: Optional[str] = None
    approved: bool = False
    approval_by: Optional[str] = None
    execution_time_ms: float = 0.0
    trace_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])


class SchemaValidator:
    """
    第一层防护: Schema硬校验
    
    面试要点:
    - 在Schema里写清楚'负面描述'
    - 用Pydantic模型核对，类型不对直接打回
    - 关键参数做业务兜底
    """
    
    def __init__(self):
        self.schemas: Dict[str, Dict] = {}
    
    def register_tool(self, name: str, schema: Dict):
        """注册工具Schema"""
        self.schemas[name] = schema
    
    def validate(self, tool_name: str, parameters: Dict) -> Dict:
        """
        校验参数
        
        返回: {"valid": bool, "errors": List[str]}
        """
        if tool_name not in self.schemas:
            return {"valid": False, "errors": [f"未知工具: {tool_name}"]}
        
        schema = self.schemas[tool_name]
        errors = []
        
        # 检查必填参数
        required = schema.get("required", [])
        for param in required:
            if param not in parameters:
                errors.append(f"缺少必填参数: {param}")
        
        # 检查参数类型
        properties = schema.get("properties", {})
        for param, value in parameters.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"参数'{param}'应为字符串，实际是{type(value).__name__}")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"参数'{param}'应为数字，实际是{type(value).__name__}")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"参数'{param}'应为整数，实际是{type(value).__name__}")
                
                # 检查枚举值
                enum_values = properties[param].get("enum")
                if enum_values and value not in enum_values:
                    errors.append(f"参数'{param}'的值'{value}'不在允许范围内: {enum_values}")
                
                # 检查范围
                minimum = properties[param].get("minimum")
                if minimum is not None and value < minimum:
                    errors.append(f"参数'{param}'的值{value}小于最小值{minimum}")
                
                maximum = properties[param].get("maximum")
                if maximum is not None and value > maximum:
                    errors.append(f"参数'{param}'的值{value}大于最大值{maximum}")
        
        return {"valid": len(errors) == 0, "errors": errors}


class PermissionGuard:
    """
    第二层防护: 权限控制
    
    面试要点:
    - 最小权限原则
    - 高风险操作需人工确认
    - 操作分级
    """
    
    def __init__(self):
        self.tool_permissions: Dict[str, RiskLevel] = {}
        self.approval_callbacks: Dict[RiskLevel, Callable] = {}
    
    def register_tool_risk(self, tool_name: str, level: RiskLevel):
        """注册工具风险等级"""
        self.tool_permissions[tool_name] = level
    
    def set_approval_callback(self, level: RiskLevel, callback: Callable):
        """设置审批回调"""
        self.approval_callbacks[level] = callback
    
    def check_permission(self, tool_name: str, parameters: Dict) -> Dict:
        """
        检查权限
        
        返回: {"allowed": bool, "requires_approval": bool, "reason": str}
        """
        risk = self.tool_permissions.get(tool_name, RiskLevel.LOW)
        
        # 只读操作自动通过
        if risk == RiskLevel.LOW:
            return {"allowed": True, "requires_approval": False, "reason": "只读操作"}
        
        # 中风险操作，记录日志
        if risk == RiskLevel.MEDIUM:
            return {"allowed": True, "requires_approval": False, "reason": "中风险，已记录"}
        
        # 高风险操作，需要审批
        if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return {
                "allowed": True,
                "requires_approval": True,
                "reason": f"{risk.value}风险操作需要确认"
            }
        
        return {"allowed": False, "requires_approval": False, "reason": "未知工具"}
    
    def request_approval(self, tool_name: str, parameters: Dict, user: str = "system") -> bool:
        """请求审批（模拟）"""
        risk = self.tool_permissions.get(tool_name, RiskLevel.LOW)
        
        if risk in self.approval_callbacks:
            return self.approval_callbacks[risk](tool_name, parameters, user)
        
        # 默认审批策略
        if risk == RiskLevel.CRITICAL:
            print(f"  ⚠️ CRITICAL操作 '{tool_name}' 需要人工确认")
            return False  # 默认拒绝
        return True


class AuditLogger:
    """
    第三层防护: 审计追踪
    
    面试要点:
    - 所有操作记录完整日志
    - 支持一键重现
    - 异常检测
    """
    
    def __init__(self):
        self.logs: List[ToolCall] = []
        self.alert_threshold = 10  # 10分钟内调用次数阈值
    
    def log_call(self, call: ToolCall):
        """记录调用"""
        self.logs.append(call)
        
        # 异常检测
        self._check_anomaly(call)
    
    def _check_anomaly(self, call: ToolCall):
        """异常检测"""
        # 检查高频调用
        recent_calls = [
            log for log in self.logs
            if log.tool_name == call.tool_name
            and call.timestamp - log.timestamp < 600  # 10分钟
        ]
        
        if len(recent_calls) > self.alert_threshold:
            print(f"  🚨 异常告警: 工具'{call.tool_name}' 10分钟内被调用{len(recent_calls)}次")
    
    def get_audit_trail(self, trace_id: str) -> Optional[ToolCall]:
        """获取审计轨迹"""
        for log in self.logs:
            if log.trace_id == trace_id:
                return log
        return None
    
    def generate_report(self) -> Dict:
        """生成审计报告"""
        total = len(self.logs)
        errors = sum(1 for log in self.logs if log.error)
        critical = sum(1 for log in self.logs if log.risk_level == RiskLevel.CRITICAL)
        
        return {
            "total_calls": total,
            "error_count": errors,
            "error_rate": errors / total if total > 0 else 0,
            "critical_operations": critical,
            "unique_tools": len(set(log.tool_name for log in self.logs))
        }


class SafeToolExecutor:
    """
    安全工具执行器 - 三层防护整合
    
    使用流程:
    1. SchemaValidator: 参数格式校验
    2. PermissionGuard: 权限检查
    3. AuditLogger: 审计记录
    """
    
    def __init__(self):
        self.validator = SchemaValidator()
        self.guard = PermissionGuard()
        self.audit = AuditLogger()
        self.tools: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, func: Callable, schema: Dict, 
                     risk_level: RiskLevel = RiskLevel.LOW):
        """注册工具"""
        self.tools[name] = func
        self.validator.register_tool(name, schema)
        self.guard.register_tool_risk(name, risk_level)
    
    def execute(self, tool_name: str, parameters: Dict, 
                user: str = "system", dry_run: bool = False) -> Dict:
        """
        安全执行工具
        
        面试要点: 所有写操作必须支持dry-run模式
        """
        print(f"\n🔧 执行工具: {tool_name}")
        print(f"   参数: {json.dumps(parameters, ensure_ascii=False)}")
        
        # 第一层: Schema校验
        validation = self.validator.validate(tool_name, parameters)
        if not validation["valid"]:
            print(f"   ❌ Schema校验失败: {validation['errors']}")
            return {
                "success": False,
                "error": f"参数校验失败: {validation['errors']}"
            }
        print(f"   ✅ Schema校验通过")
        
        # 第二层: 权限检查
        permission = self.guard.check_permission(tool_name, parameters)
        if not permission["allowed"]:
            print(f"   ❌ 权限检查失败: {permission['reason']}")
            return {
                "success": False,
                "error": f"权限不足: {permission['reason']}"
            }
        
        # 需要审批
        if permission["requires_approval"]:
            approved = self.guard.request_approval(tool_name, parameters, user)
            if not approved:
                print(f"   ❌ 审批被拒绝")
                return {
                    "success": False,
                    "error": "操作需要审批，未通过"
                }
            print(f"   ✅ 审批通过")
        
        # Dry-run模式
        if dry_run:
            print(f"   📝 Dry-run模式，不实际执行")
            return {
                "success": True,
                "result": "[Dry-run] 操作预览通过",
                "dry_run": True
            }
        
        # 第三层: 执行并审计
        start_time = time.time()
        call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            risk_level=self.guard.tool_permissions.get(tool_name, RiskLevel.LOW),
            approved=permission.get("requires_approval", False)
        )
        
        try:
            result = self.tools[tool_name](**parameters)
            call.result = str(result)
            call.execution_time_ms = (time.time() - start_time) * 1000
            print(f"   ✅ 执行成功: {str(result)[:80]}...")
            
        except Exception as e:
            call.error = str(e)
            call.execution_time_ms = (time.time() - start_time) * 1000
            print(f"   ❌ 执行失败: {str(e)}")
        
        self.audit.log_call(call)
        
        return {
            "success": call.error is None,
            "result": call.result,
            "error": call.error,
            "trace_id": call.trace_id,
            "execution_time_ms": call.execution_time_ms
        }
    
    def get_audit_report(self) -> Dict:
        """获取审计报告"""
        return self.audit.generate_report()


# ============ 演示工具 ============

def query_stock(stock_code: str) -> str:
    """查询股票信息 (只读)"""
    return f"股票{stock_code}: 价格45.2元, 涨幅2.3%"


def update_portfolio(stock_code: str, shares: int) -> str:
    """更新持仓 (中风险)"""
    return f"更新持仓: {stock_code} {shares}股"


def delete_order(order_id: str) -> str:
    """删除订单 (高风险)"""
    return f"订单{order_id}已删除"


def transfer_funds(to_account: str, amount: float) -> str:
    """转账 (CRITICAL)"""
    return f"转账{amount}元到账户{to_account}"


def demo():
    """演示三层防护"""
    
    executor = SafeToolExecutor()
    
    # 注册工具（带风险等级）
    executor.register_tool(
        "query_stock", query_stock,
        schema={
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "股票代码，如'002463'"}
            },
            "required": ["stock_code"]
        },
        risk_level=RiskLevel.LOW
    )
    
    executor.register_tool(
        "update_portfolio", update_portfolio,
        schema={
            "type": "object",
            "properties": {
                "stock_code": {"type": "string"},
                "shares": {"type": "integer", "minimum": 0}
            },
            "required": ["stock_code", "shares"]
        },
        risk_level=RiskLevel.MEDIUM
    )
    
    executor.register_tool(
        "delete_order", delete_order,
        schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        },
        risk_level=RiskLevel.HIGH
    )
    
    executor.register_tool(
        "transfer_funds", transfer_funds,
        schema={
            "type": "object",
            "properties": {
                "to_account": {"type": "string"},
                "amount": {"type": "number", "minimum": 0.01}
            },
            "required": ["to_account", "amount"]
        },
        risk_level=RiskLevel.CRITICAL
    )
    
    print("=" * 60)
    print("工具安全执行器 - 三层防护演示")
    print("=" * 60)
    
    # 场景1: 正常查询（低风险，自动通过）
    print("\n📋 场景1: 正常查询")
    result = executor.execute("query_stock", {"stock_code": "002463"})
    
    # 场景2: 参数错误（Schema校验拦截）
    print("\n📋 场景2: 参数错误")
    result = executor.execute("query_stock", {"stock_code": 12345})  # 类型错误
    
    # 场景3: 更新持仓（中风险，记录日志）
    print("\n📋 场景3: 更新持仓")
    result = executor.execute("update_portfolio", {"stock_code": "002463", "shares": 100})
    
    # 场景4: 删除订单（高风险，需要审批）
    print("\n📋 场景4: 删除订单（需要审批）")
    result = executor.execute("delete_order", {"order_id": "ORD-20240503-001"})
    
    # 场景5: 转账（CRITICAL，dry-run预览）
    print("\n📋 场景5: 转账（Dry-run预览）")
    result = executor.execute(
        "transfer_funds", 
        {"to_account": "622202123456789", "amount": 10000.00},
        dry_run=True
    )
    
    # 审计报告
    print("\n" + "=" * 60)
    print("📊 审计报告:")
    report = executor.get_audit_report()
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo()
