#!/usr/bin/env python3
"""
Computer Use Agent - 浏览器自动化
P3: 基于 Playwright 的浏览器操作 Agent

功能:
- 自动浏览网页
- 点击、输入、滚动
- 截图分析
- 数据抓取
- 表单填写

架构:
- Vision + Action Loop
- LLM 决策下一步操作
- Playwright 执行操作
"""
import os
import json
import base64
import asyncio
from typing import Dict, Any, List, Optional
from io import BytesIO
from datetime import datetime

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
except ImportError:
    print("请先安装 Playwright: pip install playwright && playwright install")
    raise

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    print("请先安装 LangChain")
    raise


# ============ 配置 ============

SYSTEM_PROMPT = """你是浏览器自动化助手。你可以控制浏览器执行各种操作。

可用操作:
1. navigate(url) - 导航到指定URL
2. click(selector) - 点击元素
3. type(selector, text) - 输入文本
4. scroll(direction, amount) - 滚动页面
5. screenshot() - 截图
6. extract(selector) - 提取元素文本
7. wait(seconds) - 等待
8. done(answer) - 完成任务

规则:
- 每次只执行一个操作
- 操作后等待页面加载完成
- 如果操作失败，尝试替代方案
- 截图后分析页面状态决定下一步
- 最终必须调用 done() 结束

输出格式（JSON）:
{
    "thought": "当前思考",
    "action": "操作名称",
    "params": {"参数名": "参数值"}
}"""


class BrowserAgent:
    """浏览器自动化 Agent"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=0,
        )
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.history: List[Dict[str, Any]] = []
        self.max_steps = 20
    
    async def start(self):
        """启动浏览器"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        )
        self.page = await self.context.new_page()
        print("✅ 浏览器已启动")
    
    async def stop(self):
        """关闭浏览器"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("🛑 浏览器已关闭")
    
    async def screenshot(self) -> str:
        """截图并转为 base64"""
        screenshot = await self.page.screenshot()
        return base64.b64encode(screenshot).decode()
    
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行浏览器操作"""
        result = {"success": True, "data": None}
        
        try:
            if action == "navigate":
                await self.page.goto(params["url"], wait_until="networkidle")
                result["data"] = f"已导航到 {params['url']}"
            
            elif action == "click":
                await self.page.click(params["selector"])
                await self.page.wait_for_load_state("networkidle")
                result["data"] = f"已点击 {params['selector']}"
            
            elif action == "type":
                await self.page.fill(params["selector"], params["text"])
                result["data"] = f"已在 {params['selector']} 输入文本"
            
            elif action == "scroll":
                direction = params.get("direction", "down")
                amount = params.get("amount", 500)
                if direction == "down":
                    await self.page.evaluate(f"window.scrollBy(0, {amount})")
                else:
                    await self.page.evaluate(f"window.scrollBy(0, -{amount})")
                result["data"] = f"已{'向下' if direction == 'down' else '向上'}滚动 {amount}px"
            
            elif action == "screenshot":
                b64 = await self.screenshot()
                result["data"] = b64
            
            elif action == "extract":
                elements = await self.page.query_selector_all(params["selector"])
                texts = []
                for el in elements[:10]:  # 最多提取10个
                    text = await el.text_content()
                    if text:
                        texts.append(text.strip())
                result["data"] = texts
            
            elif action == "wait":
                await asyncio.sleep(params.get("seconds", 1))
                result["data"] = f"等待 {params.get('seconds', 1)} 秒"
            
            elif action == "done":
                result["data"] = params.get("answer", "任务完成")
                result["done"] = True
            
            else:
                result["success"] = False
                result["data"] = f"未知操作: {action}"
        
        except Exception as e:
            result["success"] = False
            result["data"] = str(e)
        
        return result
    
    async def think(self, task: str, screenshot_b64: str = None, last_result: str = None) -> Dict[str, Any]:
        """LLM 决策下一步"""
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # 构建上下文
        context = f"任务: {task}\n"
        if last_result:
            context += f"上一步结果: {last_result}\n"
        
        messages.append(HumanMessage(content=[
            {"type": "text", "text": context},
        ]))
        
        # 如果有截图，加入 vision
        if screenshot_b64:
            messages[-1].content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            })
        
        response = await self.llm.ainvoke(messages)
        
        # 解析 JSON
        try:
            content = response.content
            # 提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            decision = json.loads(content.strip())
            return decision
        except Exception as e:
            print(f"解析失败: {e}")
            return {"thought": "解析错误", "action": "done", "params": {"answer": "无法解析决策"}}
    
    async def run(self, task: str) -> str:
        """
        执行任务
        
        Vision-Action Loop:
        1. 截图
        2. LLM 分析并决策
        3. 执行操作
        4. 重复直到完成
        """
        print(f"\n🎯 任务: {task}")
        print("=" * 50)
        
        await self.start()
        
        try:
            for step in range(self.max_steps):
                print(f"\n--- 步骤 {step + 1} ---")
                
                # 截图
                screenshot_b64 = await self.screenshot()
                
                # LLM 决策
                decision = await self.think(
                    task,
                    screenshot_b64=screenshot_b64,
                    last_result=self.history[-1]["result"] if self.history else None,
                )
                
                print(f"🤖 思考: {decision.get('thought', '')}")
                print(f"🔧 操作: {decision.get('action')}({decision.get('params', {})})")
                
                # 执行操作
                result = await self.execute_action(
                    decision["action"],
                    decision.get("params", {}),
                )
                
                print(f"📊 结果: {str(result['data'])[:100]}...")
                
                # 记录历史
                self.history.append({
                    "step": step + 1,
                    "thought": decision.get("thought"),
                    "action": decision["action"],
                    "params": decision.get("params"),
                    "result": result["data"],
                    "success": result["success"],
                })
                
                # 检查是否完成
                if result.get("done") or decision["action"] == "done":
                    print(f"\n✅ 任务完成: {result['data']}")
                    return result["data"]
                
                if not result["success"]:
                    print(f"⚠️ 操作失败: {result['data']}")
            
            print(f"\n⚠️ 达到最大步数 ({self.max_steps})")
            return "任务未完成，达到最大步数限制"
            
        finally:
            await self.stop()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.history


# ============ 专用工具 ============

class StockDataScraper:
    """股票数据抓取器"""
    
    def __init__(self):
        self.agent = BrowserAgent(headless=True)
    
    async def scrape_eastmoney(self, stock_code: str) -> Dict[str, Any]:
        """从东方财富抓取股票数据"""
        task = f"""
        请访问东方财富网，搜索股票 {stock_code}，提取以下信息：
        1. 股票名称
        2. 当前价格
        3. 涨跌幅
        4. 成交量
        5. 市值
        
        步骤:
        1. 导航到 https://quote.eastmoney.com/{stock_code}.html
        2. 等待页面加载
        3. 提取页面上的关键数据
        4. 返回结构化数据
        """
        
        result = await self.agent.run(task)
        return {"stock_code": stock_code, "raw_result": result}
    
    async def scrape_news(self, keyword: str, count: int = 5) -> List[str]:
        """抓取新闻"""
        task = f"""
        请搜索关键词 "{keyword}" 的最新新闻，提取前 {count} 条新闻标题。
        
        步骤:
        1. 导航到搜索引擎
        2. 搜索关键词
        3. 提取新闻标题
        4. 返回列表
        """
        
        result = await self.agent.run(task)
        return [result]  # 简化处理


# ============ 演示 ============

async def demo_browser_agent():
    """演示浏览器 Agent"""
    print("=" * 60)
    print("Computer Use Agent - 浏览器自动化演示")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 需要 OPENAI_API_KEY")
        return
    
    agent = BrowserAgent(headless=False)  # 有头模式便于观察
    
    task = "请访问百度，搜索 '贵州茅台 股价'，提取搜索结果中的股价信息"
    
    result = await agent.run(task)
    
    print(f"\n📊 最终结果:\n{result}")
    
    # 打印历史
    print(f"\n📜 执行历史 ({len(agent.get_history())} 步):")
    for h in agent.get_history():
        print(f"  {h['step']}. {h['action']} -> {'✅' if h['success'] else '❌'}")


async def demo_scraper():
    """演示数据抓取"""
    print("\n" + "=" * 60)
    print("股票数据抓取演示")
    print("=" * 60)
    
    scraper = StockDataScraper()
    
    result = await scraper.scrape_eastmoney("600519")
    print(f"\n📊 抓取结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")


async def main():
    await demo_browser_agent()
    # await demo_scraper()


if __name__ == "__main__":
    asyncio.run(main())
