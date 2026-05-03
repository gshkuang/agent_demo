#!/bin/bash
# 服务器环境安装脚本

set -e

echo "=== 开始安装 verl 训练环境 ==="

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "安装 Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
fi

# 创建环境
echo "创建 conda 环境..."
conda create -n verl python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl

# 安装 PyTorch (CUDA 12.1)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 verl
echo "安装 verl..."
cd ~
if [ ! -d "verl" ]; then
    git clone https://github.com/verl-project/verl.git
fi
cd verl
pip install -e .

# 安装 vllm
echo "安装 vLLM..."
pip install vllm

# 安装其他依赖
echo "安装评估依赖..."
pip install transformers datasets tqdm

echo "=== 环境安装完成 ==="
echo "请运行: conda activate verl"
