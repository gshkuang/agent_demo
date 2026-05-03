# Verl GSM8K 训练项目

## 文件说明

| 文件 | 说明 |
|------|------|
| `verl_gsm8k_test.sh` | 训练脚本（3090单卡优化版） |
| `verl_gsm8k_eval.py` | 模型评估脚本 |
| `verl_compare_results.py` | 训练前后效果对比脚本 |

## 快速开始

### 1. 环境准备

```bash
conda create -n verl python=3.11 -y
conda activate verl

# 安装 verl
git clone https://github.com/verl-project/verl.git
cd verl
pip install -e .
pip install vllm

# 安装评估依赖
pip install transformers datasets tqdm
```

### 2. 准备数据

```bash
cd verl/examples/data_preprocess
python3 gsm8k.py --local_save_dir ~/data/gsm8k
```

### 3. 基线评估（训练前）

```bash
python verl_gsm8k_eval.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_file baseline_results.json \
    --max_samples 100
```

### 4. 训练

```bash
bash verl_gsm8k_test.sh
```

### 5. 训练后评估

```bash
python verl_gsm8k_eval.py \
    --model_path ./output/gsm8k_1.5b/checkpoint-last \
    --output_file trained_results.json \
    --max_samples 100
```

### 6. 对比结果

```bash
python verl_compare_results.py baseline_results.json trained_results.json
```

## 服务器信息

- **IP**: 100.109.43.83
- **GPU**: NVIDIA 3090 24GB x 1
- **预估训练时间**: 30-60 分钟/epoch

## 注意事项

1. 训练前确保 CUDA 可用：`nvidia-smi`
2. 监控显存使用，避免 OOM
3. 如需调整 batch_size，修改 `verl_gsm8k_test.sh`
