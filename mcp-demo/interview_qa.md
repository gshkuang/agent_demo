# MCP 面试问题与解答

## 基础概念

### Q1: 什么是MCP？为什么需要它？

**答:** MCP (Model Context Protocol) 是Anthropic提出的开放协议，让AI Agent能安全地连接外部工具和数据源。

**为什么需要：**
- 以前每个AI应用都要单独集成API（Slack、GitHub、数据库等），重复造轮子
- MCP像"AI的USB-C"，统一接口，一次开发到处使用
- Agent可以动态发现和使用工具，无需硬编码

**对比：**
| 方式 | 缺点 | MCP优势 |
|------|------|---------|
| 直接调用API | 每个API格式不同，Agent需要硬编码 | 统一协议，自动发现 |
| Function Calling | 工具定义在代码里，新增需改代码 | 服务化注册，动态发现 |
| Plugin | 平台锁定（OpenAI Plugin只能OpenAI用） | 开放协议，跨平台 |

---

### Q2: MCP的核心架构是什么？

**答:** 三部分：

```
┌─────────────┐     MCP协议      ┌─────────────┐     任意协议     ┌─────────────┐
│   AI Agent   │ ◄──────────────► │ MCP Server  │ ◄──────────────► │  外部服务   │
│  (Claude等)  │   (stdio/SSE)   │  (工具封装)  │  (HTTP/DB等)   │ (股票API等) │
└─────────────┘                  └─────────────┘                └─────────────┘
       ▲
       │ 发现工具列表
       │ 调用工具
       ▼
┌─────────────┐
│  MCP Host   │
│ (Claude Desktop│
│  或你的应用)  │
└─────────────┘
```

- **Host**: 运行AI的应用（如Claude Desktop、你的Agent程序）
- **Client**: Host内的MCP客户端，管理连接
- **Server**: 提供工具的服务，通过MCP协议暴露能力

---

### Q3: MCP和Function Calling有什么区别？

**答:** 

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| 工具定义位置 | 代码里硬编码 | 独立服务，动态发现 |
| 新增工具 | 改代码、重启 | 启动新Server即可 |
| 跨平台 | 绑定特定模型 | 开放协议，通用 |
| 适用场景 | 内置固定功能 | 外部扩展生态 |

**关系：** MCP *使用* Function Calling。MCP Server把工具描述传给Agent，Agent用Function Calling格式调用。

---

## 实现细节

### Q4: 如何实现一个MCP Server？

**答:** 核心步骤（Python示例）：

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

# 1. 创建Server
app = Server("my-server")

# 2. 定义工具列表
@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_stock_price",
            description="获取股票价格",
            inputSchema={"type": "object", "properties": {"code": {"type": "string"}}}
        )
    ]

# 3. 实现工具逻辑
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_stock_price":
        price = await fetch_price(arguments["code"])
        return [TextContent(type="text", text=f"价格: {price}")]

# 4. 启动（stdio或SSE）
app.run(transport="stdio")
```

**关键点：**
- 工具描述要清晰（Agent靠描述选择工具）
- 输入用JSON Schema定义
- 输出用标准Content类型

---

### Q5: MCP支持哪些传输方式？怎么选？

**答:** 两种：

| 方式 | 适用场景 | 特点 |
|------|---------|------|
| **stdio** | 本地工具（如文件系统、数据库） | 子进程通信，简单安全 |
| **SSE** | 远程服务（如Web API） | HTTP流，可跨网络 |

**选择：**
- 本地脚本/命令行工具 → stdio
- Web服务/云API → SSE

---

### Q6: Agent怎么决定调用哪个工具？

**答:** 流程：

```
1. Agent发送 tools/list 请求给MCP Server
2. Server返回所有工具的名称、描述、参数格式
3. Agent把工具描述放入System Prompt
4. LLM根据用户问题和工具描述，决定调用哪个
5. Agent发送 tools/call 请求，携带参数
6. Server执行，返回结果
7. Agent把结果放入上下文，生成最终回答
```

**关键：** 工具描述的`description`至关重要，直接决定LLM的选择准确性。

---

## 实际应用

### Q7: 在你的项目中怎么用的MCP？

**答:** （结合本Demo回答）

**场景：** 股票分析Agent需要查询实时数据

**实现：**
1. 创建`StockAnalyzer MCP Server`，暴露3个工具：
   - `get_stock_info` - 查询单只股票
   - `compare_stocks` - 多股对比
   - `analyze_valuation` - 估值分析

2. Agent通过MCP协议发现这些工具

3. 用户问"茅台ROE多少？"时：
   - Agent识别需要查询数据
   - 从工具列表找到`get_stock_info`
   - 构造参数`{"code": "600519"}`
   - 调用工具，获取结果
   - 生成自然语言回答

**优势：** 股票分析逻辑封装在Server里，Agent只负责决策和生成，解耦清晰。

---

### Q8: MCP的安全性怎么保证？

**答:** 多层防护：

1. **权限控制**: Server可以限制哪些工具暴露、参数范围
2. **用户确认**: 敏感操作（如交易）可以让用户确认后再执行
3. **输入校验**: JSON Schema校验参数，防止注入
4. **沙箱隔离**: stdio方式下Server是独立进程，崩溃不影响Host
5. **审计日志**: 所有工具调用可记录，便于追溯

---

### Q9: MCP和LangChain的Tools有什么区别？

**答:** 

| 维度 | LangChain Tools | MCP |
|------|----------------|-----|
| 架构 | 库内函数 | 独立服务 |
| 语言绑定 | Python/JS为主 | 语言无关 |
| 发现机制 | 代码import | 协议自动发现 |
| 生态 | LangChain生态 | 跨框架通用 |

**关系：** 可以共存。LangChain可以集成MCP Client，把MCP Server当作LangChain Tool使用。

---

## 进阶问题

### Q10: 如果工具调用失败怎么办？

**答:** 错误处理策略：

1. **重试**: 网络问题自动重试3次
2. **降级**: 主API失败时切换备用源
3. **告知用户**: 明确说明无法获取数据
4. **日志记录**: 记录错误便于排查

```python
@app.call_tool()
async def call_tool(name, arguments):
    try:
        result = await handler(**arguments)
        return [TextContent(text=result)]
    except Exception as e:
        return [TextContent(text=f"错误: {str(e)}")]
```

---

### Q11: 工具很多时怎么优化？

**答:** 

1. **分层设计**: 按领域拆分多个Server（股票Server、天气Server）
2. **动态加载**: 根据对话上下文只加载相关Server
3. **缓存**: 工具列表和结果缓存，减少重复调用
4. **描述优化**: 工具描述要精准，帮助LLM快速选择

---

### Q12: MCP的未来发展方向？

**答:** 

1. **更多传输方式**: gRPC、WebSocket等
2. **标准化生态**: 类似Docker Hub的MCP Server市场
3. **安全增强**: 细粒度权限、OAuth集成
4. **多模态**: 支持图片、音频等输入输出
5. **与Agent框架深度集成**: AutoGen、LangGraph等原生支持
