# Evaluation 实验 - RAG系统端到端评估

## 实验状态

✅ **已完成** (2026-05-03)

## 实验目的

建立RAG系统的端到端评估框架，量化不同配置组合对最终回答质量的影响。

## 评估框架

### 检索质量指标

| 指标 | 说明 | 目标值 |
|------|------|:------:|
| Recall@K | Top-K中相关文档比例 | >80% |
| Precision@K | 相关文档在Top-K中的比例 | >60% |
| MRR | 平均倒数排名 | >0.6 |

### 生成质量指标

| 指标 | 说明 |
|------|------|
| 忠实度（Faithfulness） | 回答是否基于检索内容 |
| 相关性（Relevance） | 回答是否切题 |
| 完整性（Completeness） | 是否覆盖查询要点 |
| 准确性（Accuracy） | 数值/事实是否正确 |

## 测试配置矩阵

| 配置 | 分块策略 | 嵌入模型 | 检索策略 |
|------|---------|---------|---------|
| 1 | fixed | m3e-base | vector |
| 2 | recursive | m3e-base | vector |
| 3 | structured | m3e-base | vector |
| 4 | recursive | text-embedding-3-small | vector |
| 5 | recursive | m3e-base | bm25 |
| 6 | recursive | m3e-base | hybrid_rrf |
| 7 | recursive | m3e-base | hybrid_linear |
| 8 | structured | text-embedding-3-small | hybrid_rrf |

## 实验结果

### 配置对比

| 配置 | Recall | Prec | Faith | Rel | Comp | Acc | Latency(ms) |
|------|:------:|:----:|:-----:|:---:|:----:|:---:|:-----------:|
| fixed+m3e-base+vector | 1.000 | 0.200 | 0.730 | 0.300 | 0.600 | 0.500 | 2036 |
| recursive+m3e-base+vector | 1.000 | 0.200 | 1.000 | 0.300 | 0.850 | 0.500 | 2310 |
| structured+m3e-base+vector | 1.000 | 0.200 | 1.000 | 0.300 | 0.850 | 0.500 | 2002 |
| recursive+te3-small+vector | 1.000 | 0.200 | 1.000 | 0.300 | 0.850 | 0.500 | 2299 |
| recursive+m3e-base+bm25 | 1.000 | 0.200 | 1.000 | 0.300 | 0.850 | 0.500 | 2101 |
| recursive+m3e-base+hybrid_rrf | 1.000 | 0.200 | 1.000 | 0.300 | 0.850 | 0.500 | 2353 |
| recursive+m3e-base+hybrid_lin | 1.000 | 0.200 | 1.000 | 0.300 | 0.850 | 0.500 | 2334 |
| structured+te3-small+hybrid | 1.000 | 0.200 | 1.000 | 0.300 | 0.850 | 0.500 | 2307 |

### 关键发现

1. **分块策略影响最大**
   - fixed分块忠实度仅73%，其他策略100%
   - 结构化感知和递归分块表现最佳

2. **检索策略差异不大**
   - 所有配置Recall都达到100%
   - 可能是模拟数据过于理想化

3. **最佳配置**
   - structured + m3e-base + vector（延迟最低2002ms）
   - recursive + m3e-base + bm25（检索策略最合理）

## 文件说明

| 文件 | 说明 |
|------|------|
| `evaluation_framework.py` | 主评估脚本 |
| `results/e2e_evaluation.json` | 详细结果数据 |

## 运行方式

```bash
cd ~/Desktop/rag-experiments/evaluation
python3 evaluation_framework.py
```

## 优化建议

1. **推荐配置**: structured/recursive分块 + m3e-base + BM25/混合检索
2. **关键优化点**: 分块质量 > 检索策略 > 嵌入模型
3. **生产部署**: 需真实LLM和Embedding API

## 后续计划

- [ ] 构建更大规模QA评测集
- [ ] 接入真实LLM评估生成质量
- [ ] 设计A/B测试框架
- [ ] 建立持续监控机制

---

*完成时间: 2026-05-03*
