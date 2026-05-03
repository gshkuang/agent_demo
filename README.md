# agent_demo - AI Agent实验项目

## 项目概述

本项目是AI Agent实验集合，包含多个子项目。当前主要包含RAG（检索增强生成）实验，后续将扩展其他Agent能力测试。

## 项目结构

```
agent_demo/
├── README.md                  # 本文件
├── data/                      # 数据目录
│   ├── analysis/              # 分析文档（Markdown）
│   │   ├── 5stocks-beibeixia-maomao-analysis_20260421.md
│   │   ├── blockbeats-skill-test_20260429.md
│   │   ├── sector_rotation_20260418_review.md
│   │   └── 四虾回测系统-数据爬取与回测_2026-05-02.md
│   ├── financial/             # 金融数据
│   │   └── mx_data/           # 东方财富数据
│   │       └── output/        # 原始数据文件（JSON/Excel/txt）
│   └── raw/                   # 原始数据（待处理）
└── rag-experiments/           # RAG实验项目
    ├── README.md
    ├── chunking/              # 文本分块实验
    │   ├── README.md
    │   ├── rag_chunking_test.py
    │   ├── rag_chunking_test_v2.py
    │   ├── RAG_CHUNKING_TEST_REPORT.md
    │   ├── chunking_test_results.json
    │   └── chunking_test_results_v2.json
    ├── embedding/             # 向量嵌入实验
    │   ├── README.md
    │   ├── embedding_benchmark.py
    │   └── results/
    │       └── benchmark_results.json
    ├── retrieval/             # 检索策略实验
    │   ├── README.md
    │   ├── retrieval_benchmark.py
    │   └── results/
    │       └── benchmark_results.json
    └── evaluation/            # 端到端评估实验
        ├── README.md
        ├── evaluation_framework.py
        └── results/
            └── e2e_evaluation.json
```

## 数据说明

### 分析文档（data/analysis/）

| 文件 | 内容 | 来源 |
|------|------|------|
| 5stocks-beibeixia-maomao-analysis_20260421.md | 5只个股贝贝虾+毛毛分析 | Agent生成 |
| sector_rotation_20260418_review.md | 板块轮动复盘 | Agent生成 |
| 四虾回测系统-数据爬取与回测_2026-05-02.md | 回测系统设计 | Agent生成 |
| blockbeats-skill-test_20260429.md | BlockBeats技能测试 | Agent生成 |

### 金融数据（data/financial/mx_data/）

东方财富数据接口获取的原始数据，包含：

- **个股财务数据**: ROE、资产负债率、经营现金流等
- **行情数据**: 收盘价、市盈率、市净率、总市值
- **资金流向**: 主力资金流向（近5日/近10日）
- **股东信息**: 十大流通股东
- **年报数据**: 近三年营业收入、净利润

数据格式：JSON + Excel + txt（description文件）

## RAG实验结果

详见 [rag-experiments/README.md](./rag-experiments/README.md)

### 核心结论

| 模块 | 最佳方案 | 关键指标 |
|------|---------|---------|
| 文本分块 | 递归分块 / 结构化感知 | 边界质量100% |
| 向量嵌入 | m3e-base | MRR 0.203，延迟17.3ms |
| 检索策略 | BM25 | Recall@5 50% |
| 端到端评估 | structured + m3e-base + BM25 | 忠实度100% |

## 快速开始

```bash
# 进入项目目录
cd ~/Desktop/agent_demo

# 运行RAG分块测试
cd rag-experiments/chunking
python3 rag_chunking_test_v2.py

# 运行Embedding对比
cd ../embedding
python3 embedding_benchmark.py

# 运行检索策略测试
cd ../retrieval
python3 retrieval_benchmark.py

# 运行端到端评估
cd ../evaluation
python3 evaluation_framework.py
```

## 后续计划

- [ ] 接入真实Embedding API（OpenAI/BGE）
- [ ] 构建更大规模金融QA评测集
- [ ] 测试其他Agent能力（代码生成、数据分析等）
- [ ] 领域微调实验

## Git管理

```bash
# 查看状态
git status

# 提交更改
git add -A
git commit -m "描述信息"
```

## 注意事项

- `data/financial/mx_data/output/` 包含大量原始数据文件，已加入.gitignore
- 分析文档为Agent生成内容，仅供参考
- 实验脚本使用模拟数据，生产环境需接入真实API

---

*项目创建: 2026-05-03*
*维护者: Ethon*
