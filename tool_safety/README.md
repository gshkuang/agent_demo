# Tool Safety 模块

基于 LangChain BaseTool + Pydantic 的三层防护。

## 架构

```
BaseTool/StructuredTool 包装
├── Pydantic Schema校验  (参数类型、范围、枚举)
├── 权限控制              (LOW/MEDIUM/HIGH/CRITICAL)
└── 审计追踪              (AuditCallback记录所有调用)
```

## 运行

```bash
python3 safe_tool_executor.py
```

## 面试Q&A

**Q: 模型瞎传参数怎么办？**

A: Pydantic硬校验拦截，类型不对直接打回。关键参数做业务兜底。

**Q: Agent误删数据怎么防？**

A: Dry-run预览 + 高风险操作二次确认 + 最小权限账号。
