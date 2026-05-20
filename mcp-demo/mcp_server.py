#!/usr/bin/env python3
"""
MCP Server Demo - 股票分析工具服务
演示如何将自定义工具封装为MCP服务，供AI Agent调用
"""
import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

# MCP协议基础类（简化版，实际使用mcp SDK）
class MCPServer:
    """MCP服务器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, Dict] = {}
        
    def register_tool(self, name: str, description: str, parameters: Dict, handler):
        """注册工具"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        print(f"[MCP] 注册工具: {name}")
    
    async def handle_request(self, request: Dict) -> Dict:
        """处理请求"""
        method = request.get("method")
        
        if method == "tools/list":
            return self._list_tools()
        elif method == "tools/call":
            return await self._call_tool(request["params"])
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _list_tools(self) -> Dict:
        """列出所有可用工具"""
        return {
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
                for t in self.tools.values()
            ]
        }
    
    async def _call_tool(self, params: Dict) -> Dict:
        """调用工具"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return {"error": f"Tool not found: {tool_name}"}
        
        try:
            result = await self.tools[tool_name]["handler"](**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}


# ==================== 股票分析工具实现 ====================

class StockAnalyzer:
    """股票分析器 - 模拟数据"""
    
    STOCK_DB = {
        "600519": {"name": "贵州茅台", "price": 1688.88, "pe": 28.5, "pb": 8.2, "roe": 25.3},
        "000858": {"name": "五粮液", "price": 158.60, "pe": 18.2, "pb": 4.5, "roe": 20.1},
        "002594": {"name": "比亚迪", "price": 268.50, "pe": 32.1, "pb": 5.8, "roe": 15.6},
        "300750": {"name": "宁德时代", "price": 198.00, "pe": 25.8, "pb": 6.2, "roe": 18.9},
        "000333": {"name": "美的集团", "price": 65.80, "pe": 14.2, "pb": 3.1, "roe": 22.4},
    }
    
    @classmethod
    async def get_stock_info(cls, code: str) -> Dict:
        """获取股票基本信息"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        if code not in cls.STOCK_DB:
            return {"error": f"未找到股票: {code}"}
        
        info = cls.STOCK_DB[code].copy()
        info["code"] = code
        info["update_time"] = datetime.now().isoformat()
        return info
    
    @classmethod
    async def compare_stocks(cls, codes: List[str]) -> Dict:
        """对比多只股票"""
        await asyncio.sleep(0.2)
        
        results = []
        for code in codes:
            if code in cls.STOCK_DB:
                info = cls.STOCK_DB[code].copy()
                info["code"] = code
                results.append(info)
        
        if not results:
            return {"error": "未找到任何股票"}
        
        # 计算排名
        results.sort(key=lambda x: x["roe"], reverse=True)
        for i, r in enumerate(results, 1):
            r["roe_rank"] = i
        
        return {
            "stocks": results,
            "best_roe": results[0]["name"],
            "comparison_time": datetime.now().isoformat()
        }
    
    @classmethod
    async def analyze_valuation(cls, code: str) -> Dict:
        """估值分析"""
        await asyncio.sleep(0.15)
        
        if code not in cls.STOCK_DB:
            return {"error": f"未找到股票: {code}"}
        
        stock = cls.STOCK_DB[code]
        
        # 简单估值逻辑
        pe_score = max(0, min(100, (50 - stock["pe"]) * 2))
        pb_score = max(0, min(100, (10 - stock["pb"]) * 10))
        roe_score = min(100, stock["roe"] * 3)
        
        valuation = (pe_score + pb_score + roe_score) / 3
        
        return {
            "code": code,
            "name": stock["name"],
            "valuation_score": round(valuation, 1),
            "pe_score": round(pe_score, 1),
            "pb_score": round(pb_score, 1),
            "roe_score": round(roe_score, 1),
            "suggestion": "低估" if valuation > 70 else "合理" if valuation > 40 else "高估",
            "analysis_time": datetime.now().isoformat()
        }


# ==================== 创建MCP服务器 ====================

async def create_stock_mcp_server() -> MCPServer:
    """创建股票分析MCP服务器"""
    server = MCPServer("stock-analyzer")
    
    # 注册工具1: 获取股票信息
    server.register_tool(
        name="get_stock_info",
        description="获取单只股票的基本信息，包括价格、PE、PB、ROE等",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "股票代码，如600519"
                }
            },
            "required": ["code"]
        },
        handler=StockAnalyzer.get_stock_info
    )
    
    # 注册工具2: 对比股票
    server.register_tool(
        name="compare_stocks",
        description="对比多只股票的关键指标，返回排名",
        parameters={
            "type": "object",
            "properties": {
                "codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "股票代码列表，如[\"600519\",\"000858\"]"
                }
            },
            "required": ["codes"]
        },
        handler=StockAnalyzer.compare_stocks
    )
    
    # 注册工具3: 估值分析
    server.register_tool(
        name="analyze_valuation",
        description="对股票进行估值分析，给出评分和建议",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "股票代码"
                }
            },
            "required": ["code"]
        },
        handler=StockAnalyzer.analyze_valuation
    )
    
    return server


# ==================== 演示运行 ====================

async def demo():
    """演示MCP服务器的使用"""
    print("=" * 60)
    print("🚀 MCP Server Demo - 股票分析工具服务")
    print("=" * 60)
    
    # 创建服务器
    server = await create_stock_mcp_server()
    print(f"\n✅ 服务器 '{server.name}' 启动成功")
    print(f"   已注册 {len(server.tools)} 个工具\n")
    
    # 演示1: 列出工具
    print("📋 步骤1: AI Agent 发现可用工具")
    print("-" * 40)
    response = await server.handle_request({"method": "tools/list"})
    for tool in response["tools"]:
        print(f"\n🔧 {tool['name']}")
        print(f"   描述: {tool['description']}")
        print(f"   参数: {json.dumps(tool['parameters'], ensure_ascii=False)}")
    
    # 演示2: 调用工具 - 获取股票信息
    print("\n\n📊 步骤2: AI Agent 调用工具 - 获取股票信息")
    print("-" * 40)
    print("Agent思考: 用户问茅台的ROE，我需要调用get_stock_info\n")
    
    response = await server.handle_request({
        "method": "tools/call",
        "params": {
            "name": "get_stock_info",
            "arguments": {"code": "600519"}
        }
    })
    print(f"结果: {json.dumps(response['result'], ensure_ascii=False, indent=2)}")
    
    # 演示3: 调用工具 - 对比股票
    print("\n\n📈 步骤3: AI Agent 调用工具 - 对比多只股票")
    print("-" * 40)
    print("Agent思考: 用户想对比白酒和新能源龙头\n")
    
    response = await server.handle_request({
        "method": "tools/call",
        "params": {
            "name": "compare_stocks",
            "arguments": {"codes": ["600519", "000858", "002594", "300750"]}
        }
    })
    result = response['result']
    print(f"对比结果:")
    print(f"  ROE排名: {result['best_roe']} 第一")
    for stock in result['stocks']:
        print(f"  {stock['roe_rank']}. {stock['name']}({stock['code']}) ROE:{stock['roe']}%")
    
    # 演示4: 估值分析
    print("\n\n💰 步骤4: AI Agent 调用工具 - 估值分析")
    print("-" * 40)
    print("Agent思考: 用户问比亚迪是否值得买，我来做估值分析\n")
    
    response = await server.handle_request({
        "method": "tools/call",
        "params": {
            "name": "analyze_valuation",
            "arguments": {"code": "002594"}
        }
    })
    result = response['result']
    print(f"估值分析:")
    print(f"  股票: {result['name']}({result['code']})")
    print(f"  综合评分: {result['valuation_score']}/100")
    print(f"  PE评分: {result['pe_score']}, PB评分: {result['pb_score']}, ROE评分: {result['roe_score']}")
    print(f"  建议: {result['suggestion']}")
    
    print("\n" + "=" * 60)
    print("✅ Demo 完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
