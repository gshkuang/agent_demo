#!/usr/bin/env python3
"""
MCP Server - 股票分析工具服务 (FastMCP 版本)
P1: 基于官方 MCP SDK 的 Server，可被 Claude / Cursor / 任何 MCP Client 调用

功能:
- get_stock_info: 获取股票基本信息
- compare_stocks: 对比多只股票
- analyze_valuation: 估值分析
- get_market_sentiment: 市场情绪
- get_sector_analysis: 板块分析

运行: python stock_mcp_server.py
测试: mcp-inspector 或 Claude Desktop 配置
"""
import os
import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# FastMCP - 官方 MCP Python SDK
try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.server.sse import SseServerTransport
    from mcp.types import TextContent, ImageContent, EmbeddedResource
except ImportError:
    print("请先安装 mcp SDK: pip install mcp")
    raise


# ============ 数据层 ============

class StockDatabase:
    """模拟股票数据库 - 实际可替换为真实数据源"""
    
    STOCKS = {
        "600519": {"name": "贵州茅台", "price": 1688.88, "pe": 28.5, "pb": 8.2, "roe": 25.3, "sector": "白酒"},
        "000858": {"name": "五粮液", "price": 158.60, "pe": 18.2, "pb": 4.5, "roe": 20.1, "sector": "白酒"},
        "002594": {"name": "比亚迪", "price": 268.50, "pe": 32.1, "pb": 5.8, "roe": 15.6, "sector": "新能源"},
        "300750": {"name": "宁德时代", "price": 198.00, "pe": 25.8, "pb": 6.2, "roe": 18.9, "sector": "新能源"},
        "000333": {"name": "美的集团", "price": 65.80, "pe": 14.2, "pb": 3.1, "roe": 22.4, "sector": "家电"},
        "002463": {"name": "沪电股份", "price": 32.50, "pe": 25.0, "pb": 4.8, "roe": 18.5, "sector": "半导体"},
        "688981": {"name": "中芯国际", "price": 85.20, "pe": 120.0, "pb": 3.5, "roe": 5.2, "sector": "半导体"},
    }
    
    SECTOR_DATA = {
        "半导体": {"change": 2.3, "inflow": 15.0, "leaders": ["中芯国际", "沪电股份"]},
        "新能源": {"change": 1.8, "inflow": 8.0, "leaders": ["比亚迪", "宁德时代"]},
        "白酒": {"change": -0.5, "inflow": -3.0, "leaders": ["贵州茅台", "五粮液"]},
        "家电": {"change": 0.8, "inflow": 2.5, "leaders": ["美的集团"]},
    }
    
    @classmethod
    def get(cls, code: str) -> Optional[Dict]:
        return cls.STOCKS.get(code)
    
    @classmethod
    def search_by_name(cls, name: str) -> Optional[str]:
        for code, info in cls.STOCKS.items():
            if name in info["name"]:
                return code
        return None


# ============ MCP Server ============

# 创建 FastMCP 实例
mcp = FastMCP(
    "stock-analyzer",
    instructions="""
    你是专业的股票分析助手，可以通过以下工具获取股票数据：
    - get_stock_info: 获取单只股票的基本信息
    - compare_stocks: 对比多只股票的关键指标
    - analyze_valuation: 对股票进行估值分析
    - get_market_sentiment: 获取当前市场情绪
    - get_sector_analysis: 分析特定板块
    
    分析时请结合基本面数据（ROE、PE、PB）和市场情绪给出建议。
    """,
    dependencies=["mcp", "httpx"],
)


@mcp.tool()
async def get_stock_info(code: str) -> str:
    """
    获取单只股票的基本信息
    
    Args:
        code: 股票代码，如 600519
    """
    stock = StockDatabase.get(code)
    if not stock:
        # 尝试按名称搜索
        found_code = StockDatabase.search_by_name(code)
        if found_code:
            stock = StockDatabase.get(found_code)
            code = found_code
        else:
            return f"未找到股票: {code}"
    
    return json.dumps({
        "code": code,
        "name": stock["name"],
        "price": stock["price"],
        "pe": stock["pe"],
        "pb": stock["pb"],
        "roe": stock["roe"],
        "sector": stock["sector"],
        "update_time": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def compare_stocks(codes: str) -> str:
    """
    对比多只股票的关键指标
    
    Args:
        codes: 股票代码列表，逗号分隔，如 "600519,000858,002594"
    """
    code_list = [c.strip() for c in codes.split(",")]
    results = []
    
    for code in code_list:
        stock = StockDatabase.get(code)
        if stock:
            results.append({
                "code": code,
                "name": stock["name"],
                "price": stock["price"],
                "pe": stock["pe"],
                "pb": stock["pb"],
                "roe": stock["roe"],
            })
    
    if not results:
        return "未找到任何股票数据"
    
    # 按 ROE 排序
    results.sort(key=lambda x: x["roe"], reverse=True)
    
    return json.dumps({
        "stocks": results,
        "best_roe": results[0]["name"],
        "comparison_time": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def analyze_valuation(code: str) -> str:
    """
    对股票进行估值分析，给出评分和建议
    
    Args:
        code: 股票代码
    """
    stock = StockDatabase.get(code)
    if not stock:
        return f"未找到股票: {code}"
    
    # 估值评分逻辑
    pe_score = max(0, min(100, (50 - stock["pe"]) * 2))
    pb_score = max(0, min(100, (10 - stock["pb"]) * 10))
    roe_score = min(100, stock["roe"] * 3)
    
    valuation = (pe_score + pb_score + roe_score) / 3
    
    suggestion = "低估" if valuation > 70 else "合理" if valuation > 40 else "高估"
    
    return json.dumps({
        "code": code,
        "name": stock["name"],
        "valuation_score": round(valuation, 1),
        "pe_score": round(pe_score, 1),
        "pb_score": round(pb_score, 1),
        "roe_score": round(roe_score, 1),
        "suggestion": suggestion,
        "analysis": f"{stock['name']}当前PE{stock['pe']}倍，{'低于' if stock['pe'] < 25 else '高于'}行业平均，建议{suggestion}关注。",
        "analysis_time": datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_market_sentiment() -> str:
    """
    获取当前市场情绪数据
    """
    return json.dumps({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "limit_up": 45,
        "limit_down": 12,
        "northbound_inflow": 23.5,
        "sentiment": "偏乐观",
        "description": "今日涨停45家，跌停12家，北向资金净流入23.5亿，市场情绪偏乐观。半导体板块领涨。",
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_sector_analysis(sector: str) -> str:
    """
    分析特定板块的表现
    
    Args:
        sector: 板块名称，如 "半导体"、"新能源"、"白酒"
    """
    data = StockDatabase.SECTOR_DATA.get(sector)
    if not data:
        return f"暂无 {sector} 板块数据"
    
    return json.dumps({
        "sector": sector,
        "daily_change": data["change"],
        "fund_inflow": data["inflow"],
        "leaders": data["leaders"],
        "analysis": f"{sector}板块今日{'上涨' if data['change'] > 0 else '下跌'}{abs(data['change'])}%，资金{'净流入' if data['inflow'] > 0 else '净流出'}{abs(data['inflow'])}亿。",
    }, ensure_ascii=False, indent=2)


@mcp.resource("stock://{code}")
async def get_stock_resource(code: str) -> str:
    """
    MCP Resource: 股票数据资源
    可通过 stock://600519 访问
    """
    stock = StockDatabase.get(code)
    if not stock:
        return f"Stock {code} not found"
    return json.dumps(stock, ensure_ascii=False)


@mcp.prompt()
def stock_analysis_prompt(code: str) -> str:
    """
    MCP Prompt: 股票分析模板
    """
    return f"""请分析股票 {code} 的投资价值。

请按以下步骤分析：
1. 获取股票基本信息（价格、PE、PB、ROE）
2. 对比同行业其他股票
3. 进行估值分析
4. 结合市场情绪给出建议

请给出明确的买入/持有/观望建议，并说明理由。"""


# ============ 启动方式 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                       help="传输方式: stdio (默认，用于 Claude Desktop) 或 sse (用于 Web)")
    parser.add_argument("--port", type=int, default=8000, help="SSE 模式端口")
    args = parser.parse_args()
    
    if args.transport == "stdio":
        # stdio 模式 - 用于 Claude Desktop / Cursor
        print("🚀 Stock MCP Server 启动 (stdio 模式)")
        print("   工具: get_stock_info, compare_stocks, analyze_valuation")
        print("   资源: stock://{code}")
        print("   提示词: stock_analysis_prompt")
        mcp.run(transport="stdio")
    else:
        # SSE 模式 - 用于 Web 应用
        print(f"🚀 Stock MCP Server 启动 (SSE 模式, 端口 {args.port})")
        mcp.run(transport="sse", port=args.port)
