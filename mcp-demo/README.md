# MCP Demo - 股票分析工具服务

## 文件说明

| 文件 | 说明 |
|------|------|
| `mcp_server.py` | MCP服务器实现，暴露股票分析工具 |
| `mcp_client.py` | MCP客户端，AI Agent调用工具的演示 |
| `interview_qa.md` | MCP面试问题与解答 |

## 快速运行

```bash
cd ~/Desktop/agent_demo/mcp-demo

# 运行MCP服务器演示
python3 mcp_server.py

# 运行Agent调用演示
python3 mcp_client.py
```

## 架构

```
┌─────────────┐     MCP协议      ┌─────────────┐
│  AI Agent   │ ◄──────────────► │ MCP Server  │
│  (mcp_client)│   (tools/list)  │(mcp_server) │
│             │   (tools/call)  │             │
└─────────────┘                 └─────────────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │ 股票分析工具  │
                                │ (模拟数据)   │
                                └─────────────┘
```

## 工具列表

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `get_stock_info` | 获取股票基本信息 | `code`: 股票代码 |
| `compare_stocks` | 对比多只股票 | `codes`: 代码列表 |
| `analyze_valuation` | 估值分析 | `code`: 股票代码 |

## 核心概念

**MCP (Model Context Protocol)**: Anthropic提出的开放协议，让AI Agent安全连接外部工具。

**为什么用MCP:**
- 统一接口：不用为每个API写不同的集成代码
- 动态发现：Agent自动获取可用工具列表
- 解耦：工具逻辑和Agent逻辑分离

**MCP vs Function Calling:**
- Function Calling: 工具定义在代码里，硬编码
- MCP: 工具作为服务暴露，动态发现

## 面试要点

见 `interview_qa.md`，包含：
- 基础概念（什么是MCP、架构）
- 实现细节（如何写Server、传输方式）
- 实际应用（项目中怎么用）
- 进阶问题（安全、性能、未来）
