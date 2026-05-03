# Planning 模块

面试考点: ReAct框架、Reflexion自我反思、Plan-and-Execute选型

## 核心实现

### 1. ReAct Agent (`react_agent.py`)

**面试要点**: ReAct通过推理与行动交替，提升复杂任务可执行性

```
Thought → Action → Observation → Thought → ... → Final Answer
```

**关键设计**:
- 必须设置步数上限，防止成本和时延失控
- 每步回填结果，再进入下一轮推理
- 可解释、可审计、可重放

**运行**:
```bash
python planning/react_agent.py
```

### 2. Reflexion Agent (`reflexion_agent.py`)

**面试要点**: 失败后生成改进建议，存入记忆避免重复错误

```
执行 → 反思(Critique) → 改进(Improvement) → 重试
```

**关键设计**:
- 需要设置停止条件，防止无限循环
- 经验教训存入memory，下次复用
- 轻量化版本适合线上（如客服话术生成）

**运行**:
```bash
python planning/reflexion_agent.py
```

## 面试Q&A

**Q: ReAct vs Plan-and-Execute怎么选？**

A: 看任务不确定性。流程固定的内部工具用Plan（省成本）；用户开放任务用ReAct（灵活）。关键技巧：Plan的每个步骤里加'ReAct式检查点'，调用API后自动检查字段完整性。

**Q: Tree of Thoughts在线上能用吗？**

A: 重度ToT不能在线上用，但轻量化版本是杀手锏。比如客服话术生成：同时生成3条不同风格回复，用轻量评判模型选出最合适的一条。成本是单次3倍，但质量提升一个档次，ROI很高。
