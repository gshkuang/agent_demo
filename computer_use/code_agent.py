#!/usr/bin/env python3
"""
Code Execution Agent - 代码执行 Agent
P3: 可以编写、执行、调试代码的 Agent

功能:
- 编写 Python 代码解决问题
- 在沙箱环境中执行代码
- 分析执行结果
- 迭代改进代码

安全:
- 超时控制
- 资源限制
- 禁止危险操作
"""
import os
import re
import json
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    print("请先安装 LangChain")
    raise


# ============ 配置 ============

SYSTEM_PROMPT = """你是代码执行助手。你可以编写 Python 代码来解决问题。

规则:
1. 每次回复必须包含一个 Python 代码块
2. 代码会被自动执行，执行结果会反馈给你
3. 根据执行结果，你可以修改代码并重新执行
4. 最终给出完整的答案

代码块格式:
```python
# 你的代码
```

安全限制:
- 禁止执行系统命令 (os.system, subprocess 等)
- 禁止网络请求
- 禁止文件系统操作（除当前目录外）
- 执行超时 30 秒
"""


class CodeExecutor:
    """代码执行器 - 沙箱环境"""
    
    def __init__(self, timeout: int = 30, max_output: int = 10000):
        self.timeout = timeout
        self.max_output = max_output
        self.execution_history: List[Dict[str, Any]] = []
    
    def extract_code(self, text: str) -> Optional[str]:
        """从文本中提取 Python 代码"""
        # 匹配 ```python ... ``` 格式
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 匹配 ``` ... ``` 格式
        pattern = r'```\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    def is_safe_code(self, code: str) -> tuple[bool, str]:
        """检查代码安全性"""
        dangerous_patterns = [
            r'os\.system\s*\(',
            r'subprocess\.',
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(.*?[\"\']/(?:etc|usr|bin|sbin)',
            r'import\s+socket',
            r'urllib\.request',
            r'requests\.',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False, f"检测到危险代码模式: {pattern}"
        
        return True, "安全"
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        执行 Python 代码
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "execution_time": float,
            }
        """
        # 安全检查
        is_safe, reason = self.is_safe_code(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": f"安全拦截: {reason}",
                "execution_time": 0,
            }
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # 添加输出捕获
            wrapped_code = f'''
import sys
from io import StringIO

# 捕获输出
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = StringIO()
sys.stderr = StringIO()

try:
{chr(10).join("    " + line for line in code.split(chr(10)))}
except Exception as e:
    import traceback
    print(f"ERROR: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

# 获取输出
stdout_output = sys.stdout.getvalue()
stderr_output = sys.stderr.getvalue()

sys.stdout = old_stdout
sys.stderr = old_stderr

print("___STDOUT_START___")
print(stdout_output)
print("___STDOUT_END___")
print("___STDERR_START___")
print(stderr_output)
print("___STDERR_END___")
'''
            f.write(wrapped_code)
            temp_file = f.name
        
        try:
            # 执行代码
            start_time = datetime.now()
            result = subprocess.run(
                ["python3", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 解析输出
            output = result.stdout
            error = result.stderr
            
            # 提取捕获的输出
            stdout_match = re.search(r'___STDOUT_START___\n(.*?)\n___STDOUT_END___', output, re.DOTALL)
            stderr_match = re.search(r'___STDERR_START___\n(.*?)\n___STDERR_END___', output, re.DOTALL)
            
            if stdout_match:
                output = stdout_match.group(1)
            if stderr_match:
                error = stderr_match.group(1)
            
            # 截断输出
            if len(output) > self.max_output:
                output = output[:self.max_output] + "\n... (输出已截断)"
            
            success = result.returncode == 0 and not error.strip().startswith("ERROR:")
            
            execution_result = {
                "success": success,
                "output": output.strip(),
                "error": error.strip() if not success else "",
                "execution_time": execution_time,
            }
            
            self.execution_history.append({
                "code": code,
                "result": execution_result,
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"执行超时（{self.timeout}秒）",
                "execution_time": self.timeout,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
            }
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file)
            except:
                pass


class CodeAgent:
    """代码执行 Agent"""
    
    def __init__(self, max_iterations: int = 5):
        self.llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=0,
        )
        self.executor = CodeExecutor()
        self.max_iterations = max_iterations
        self.conversation_history: List = []
    
    async def solve(self, task: str) -> Dict[str, Any]:
        """
        解决任务
        
        Loop:
        1. LLM 生成代码
        2. 执行代码
        3. 反馈结果给 LLM
        4. 如果需要，LLM 修改代码
        5. 重复直到完成或达到最大迭代次数
        """
        print(f"\n🎯 任务: {task}")
        print("=" * 50)
        
        # 初始提示
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"请编写 Python 代码解决以下问题:\n\n{task}\n\n请直接给出代码。"),
        ]
        
        for iteration in range(self.max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            # LLM 生成代码
            response = await self.llm.ainvoke(messages)
            code = self.executor.extract_code(response.content)
            
            if not code:
                print("⚠️ 未检测到代码块，尝试直接提取...")
                code = response.content.strip()
            
            print(f"📝 生成代码:\n{code[:200]}...")
            
            # 执行代码
            result = self.executor.execute(code)
            
            print(f"{'✅' if result['success'] else '❌'} 执行结果:")
            if result["output"]:
                print(f"   输出: {result['output'][:200]}...")
            if result["error"]:
                print(f"   错误: {result['error'][:200]}...")
            print(f"   耗时: {result['execution_time']:.2f}s")
            
            # 如果成功，返回结果
            if result["success"] and result["output"]:
                return {
                    "success": True,
                    "answer": result["output"],
                    "code": code,
                    "iterations": iteration + 1,
                    "execution_time": result["execution_time"],
                }
            
            # 如果失败，反馈给 LLM
            feedback = f"""
代码执行{'成功' if result['success'] else '失败'}。

{'输出:' if result['success'] else '错误:'}
{result['output'] if result['success'] else result['error']}

请修改代码解决问题。注意:
{'- 输出为空，请确保代码有 print 输出结果' if result['success'] and not result['output'] else '- 修复上述错误'}
- 确保最终结果是完整的答案
"""
            
            messages.append(HumanMessage(content=response.content))
            messages.append(HumanMessage(content=feedback))
        
        # 达到最大迭代次数
        return {
            "success": False,
            "answer": "达到最大迭代次数，未能完成任务",
            "code": code,
            "iterations": self.max_iterations,
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.executor.execution_history


# ============ 演示 ============

async def demo_code_agent():
    """演示代码执行 Agent"""
    print("=" * 60)
    print("Code Execution Agent 演示")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ 需要 OPENAI_API_KEY")
        return
    
    agent = CodeAgent(max_iterations=3)
    
    # 任务 1: 简单计算
    task1 = "计算贵州茅台、五粮液、比亚迪、宁德时代、美的集团的 ROE 平均值和标准差。ROE分别为: 25.3%, 20.1%, 15.6%, 18.9%, 22.4%"
    
    result1 = await agent.solve(task1)
    print(f"\n📊 任务1结果:\n{result1['answer']}")
    
    # 任务 2: 数据分析
    task2 = """
    有以下股票数据，请计算每只股票的估值评分（PE越低越好，ROE越高越好）：
    贵州茅台: PE=28.5, ROE=25.3%
    五粮液: PE=18.2, ROE=20.1%
    比亚迪: PE=32.1, ROE=15.6%
    宁德时代: PE=25.8, ROE=18.9%
    美的集团: PE=14.2, ROE=22.4%
    
    评分公式: score = ROE * 3 - PE
    请输出排名和评分。
    """
    
    result2 = await agent.solve(task2)
    print(f"\n📊 任务2结果:\n{result2['answer']}")


def demo_safe_check():
    """演示安全检查"""
    print("\n" + "=" * 60)
    print("代码安全检查演示")
    print("=" * 60)
    
    executor = CodeExecutor()
    
    # 安全代码
    safe_code = """
import math
result = math.sqrt(16)
print(f"结果: {result}")
"""
    is_safe, reason = executor.is_safe_code(safe_code)
    print(f"\n安全代码:\n{safe_code}")
    print(f"检查结果: {'✅ 通过' if is_safe else '❌ 拦截'} - {reason}")
    
    # 危险代码
    dangerous_code = """
import os
os.system("rm -rf /")
"""
    is_safe, reason = executor.is_safe_code(dangerous_code)
    print(f"\n危险代码:\n{dangerous_code}")
    print(f"检查结果: {'✅ 通过' if is_safe else '❌ 拦截'} - {reason}")


async def main():
    await demo_code_agent()
    demo_safe_check()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
