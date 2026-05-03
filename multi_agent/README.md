# Multi-Agent 模块

面试考点: 多Agent协作、Manager-Worker模式、通信协议

## 核心实现

### Manager-Worker System (`manager_worker.py`)

**架构**:

```
Manager Agent
    ├── 财报分析师 (financial_analyst)
    ├── 情绪分析师 (sentiment_analyst)
    ├── 轮动分析师 (rotation_analyst)
    ├── 量化分析师 (quant_analyst)
    └── 报告撰写员 (report_writer)
```

**协作流程**:
```
Cron触发 → 财报分析 → 板块情绪 → 轮动监控 → 量化回测 → 生成报告
```

**技术实现**:
- DAG定义执行顺序和依赖关系
- ThreadPoolExecutor并行执行无依赖任务
- 共享上下文传递状态
- 错误处理: 任一Agent失败，整体流程中断

**运行**:
```bash
python multi_agent/manager_worker.py
```

## 面试Q&A

**Q: 多Agent之间怎么通信？**

A: 三种模式：
1. **层级式**: Manager分配任务，Worker通过返回值传递结果
2. **消息队列**: Redis/RabbitMQ异步通信，解耦可扩展
3. **共享Memory**: 所有Agent读写同一块状态（类似黑板模式）

**Q: 投研平台四类Agent怎么协作？**

A: 按DAG顺序执行：财报分析→板块情绪→轮动监控→量化回测。中间结果写入共享上下文，下一个Agent读取后继续。错误处理：任一Agent失败，整体流程中断并告警。
