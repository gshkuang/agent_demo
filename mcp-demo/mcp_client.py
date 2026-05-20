#!/usr/bin/env python3
"""
MCP Client Demo - AI Agent调用MCP工具
演示Agent如何发现和使用MCP服务提供的工具
"""
import json
import asyncio
from typing import List, Dict, Any

class MCPClient:
    """MCP客户端 - AI Agent侧"""
    
    def __init__(self, server_url: str = "local"):
        self.server_url = server_url
        self.available_tools: List[Dict] = []
        
    async def connect(self):
        """连接到MCP服务器，获取工具列表"""
        # 实际实现中会发送HTTP/SSE请求
        # 这里模拟从服务器获取工具列表
        print(f"[Client] 连接到MCP服务器: {self.server_url}")
        self.available_tools = await self._fetch_tools()
        print(f"[Client] 发现 {len(self.available_tools)} 个工具\n")
        
    async def _fetch_tools(self) -> List[Dict]:
        """获取工具列表（模拟）"""
        # 实际: return requests.get(f"{server_url}/tools").json()
        return [
            {
                "name": "get_stock_info",
                "description": "获取单只股票的基本信息",
                "parameters": {"code": "string"}
            },
            {
                "name": "compare_stocks", 
                "description": "对比多只股票",
                "parameters": {"codes": ["string"]}
            },
            {
                "name": "analyze_valuation",
                "description": "估值分析",
                "parameters": {"code": "string"}
            }
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """调用工具"""
        print(f"[Client] 调用工具: {tool_name}")
        print(f"[Client] 参数: {json.dumps(arguments, ensure_ascii=False)}")
        
        # 实际实现中会发送请求到服务器
        # 这里直接导入server的handler执行（演示目的）
        from mcp_server import StockAnalyzer
        
        handler = getattr(StockAnalyzer, tool_name)
        result = await handler(**arguments)
        
        print(f"[Client] 返回结果: {json.dumps(result, ensure_ascii=False)[:100]}...\n")
        return result
    
    def get_tool_descriptions(self) -> str:
        """获取工具描述，用于构建Agent Prompt"""
        descriptions = []
        for tool in self.available_tools:
            params = json.dumps(tool["parameters"], ensure_ascii=False)
            descriptions.append(
                f"- {tool['name']}: {tool['description']} (参数: {params})"
            )
        return "\n".join(descriptions)


class AIAgent:
    """
    AI Agent - 使用MCP工具解决问题
    
    核心逻辑:
    1. 接收用户问题
    2. 分析问题，决定是否需要调用工具
    3. 选择合适的工具并构造参数
    4. 获取工具结果，生成最终回答
    """
    
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
        self.conversation_history = []
        
    async def chat(self, user_input: str) -> str:
        """处理用户输入"""
        print(f"\n{'='*50}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*50}\n")
        
        # Step 1: 判断是否需要调用工具
        tool_plan = self._plan_tools(user_input)
        
        if not tool_plan:
            # 不需要工具，直接回答
            return "这个问题不需要查询数据，我可以直接回答..."
        
        # Step 2: 执行工具调用
        tool_results = []
        for step in tool_plan:
            print(f"🤖 Agent思考: {step['reasoning']}")
            result = await self.client.call_tool(step['tool'], step['args'])
            tool_results.append({
                "tool": step['tool'],
                "args": step['args'],
                "result": result
            })
        
        # Step 3: 综合结果生成回答
        answer = self._generate_answer(user_input, tool_results)
        
        print(f"🤖 Agent回答:\n{answer}\n")
        return answer
    
    def _plan_tools(self, user_input: str) -> List[Dict]:
        """
        工具规划 - 根据用户输入决定调用哪些工具
        
        实际实现中这里会用LLM做决策:
        - 把工具描述放入prompt
        - LLM输出JSON格式的调用计划
        """
        user_input = user_input.lower()
        plan = []
        
        # 简单规则匹配（演示用，实际用LLM）
        if "roe" in user_input and ("茅台" in user_input or "600519" in user_input):
            plan.append({
                "reasoning": "用户询问茅台的ROE，需要查询股票基本信息",
                "tool": "get_stock_info",
                "args": {"code": "600519"}
            })
        
        elif "对比" in user_input or "比较" in user_input:
            codes = []
            if "茅台" in user_input or "600519" in user_input:
                codes.append("600519")
            if "五粮液" in user_input or "000858" in user_input:
                codes.append("000858")
            if "比亚迪" in user_input or "002594" in user_input:
                codes.append("002594")
            if "宁德" in user_input or "300750" in user_input:
                codes.append("300750")
            
            if len(codes) >= 2:
                plan.append({
                    "reasoning": f"用户想对比{len(codes)}只股票，调用compare_stocks",
                    "tool": "compare_stocks",
                    "args": {"codes": codes}
                })
        
        elif "估值" in user_input or "值得买" in user_input or "怎么样" in user_input:
            code = None
            if "比亚迪" in user_input or "002594" in user_input:
                code = "002594"
            elif "茅台" in user_input:
                code = "600519"
            elif "美的" in user_input:
                code = "000333"
            
            if code:
                plan.append({
                    "reasoning": f"用户询问股票估值，调用analyze_valuation分析{code}",
                    "tool": "analyze_valuation",
                    "args": {"code": code}
                })
        
        return plan
    
    def _generate_answer(self, user_input: str, tool_results: List[Dict]) -> str:
        """根据工具结果生成回答"""
        if not tool_results:
            return "抱歉，我无法处理这个问题。"
        
        # 实际实现中这里会用LLM生成自然语言回答
        # 这里简单拼接结果
        parts = []
        for tr in tool_results:
            result = tr['result']
            if 'error' in result:
                parts.append(f"查询失败: {result['error']}")
                continue
            
            if tr['tool'] == 'get_stock_info':
                parts.append(
                    f"{result['name']}({result['code']})的最新数据:\n"
                    f"- 价格: ¥{result['price']}\n"
                    f"- ROE: {result['roe']}%\n"
                    f"- PE: {result['pe']}\n"
                    f"- PB: {result['pb']}"
                )
            
            elif tr['tool'] == 'compare_stocks':
                lines = ["ROE排名:"]
                for s in result['stocks']:
                    lines.append(f"  {s['roe_rank']}. {s['name']}: {s['roe']}%")
                parts.append("\n".join(lines))
            
            elif tr['tool'] == 'analyze_valuation':
                parts.append(
                    f"{result['name']}估值分析:\n"
                    f"- 综合评分: {result['valuation_score']}/100\n"
                    f"- 估值建议: {result['suggestion']}"
                )
        
        return "\n\n".join(parts)


async def demo():
    """演示Agent使用MCP工具"""
    print("=" * 60)
    print("🤖 MCP Client Demo - AI Agent 调用工具")
    print("=" * 60)
    
    # 创建客户端并连接
    client = MCPClient(server_url="http://localhost:8000")
    await client.connect()
    
    # 显示可用工具
    print("📋 Agent可用的工具:")
    print(client.get_tool_descriptions())
    
    # 创建Agent
    agent = AIAgent(client)
    
    # 场景1: 查询单只股票
    await agent.chat("贵州茅台的ROE是多少？")
    
    # 场景2: 对比多只股票
    await agent.chat("对比茅台、五粮液、比亚迪、宁德时代的ROE")
    
    # 场景3: 估值分析
    await agent.chat("比亚迪现在估值怎么样，值得买吗？")
    
    print("\n" + "=" * 60)
    print("✅ Demo 完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
