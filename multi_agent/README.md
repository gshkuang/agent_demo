# Multi-Agent 模块

基于 LangGraph Send 的 Manager-Worker 多Agent协作。

## 架构

```
Manager (分解任务)
  └── Send → Worker_1 (并行)
  └── Send → Worker_2 (并行)
  └── Send → Worker_3 (依赖完成后)
      └── Aggregator (汇总报告)
```

## 运行

```bash
python3 manager_worker.py
```

## 面试Q&A

**Q: 多Agent之间怎么通信？**

A: LangGraph的Send做动态分发，共享State传递结果。需要异步解耦时用Redis/RabbitMQ。

**Q: 投研平台四类Agent怎么协作？**

A: 按DAG顺序：财报→情绪→轮动→量化。Manager分解任务，Worker并行执行无依赖项，Aggregator汇总生成报告。
