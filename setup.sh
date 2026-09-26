#!/bin/bash

# Hentikan skrip jika ada satu perintah yang gagal
set -e

echo "🚀 [1/4] Installing system CLI tools..."
apt-get update && apt-get install -y tmux tree libz3-dev

echo "📦 [2/4] Installing Mamba dependencies & pre-compiled wheels..."

pip install --no-deps "triton>=3.5.0" "tilelang==0.1.8" "quack-kernels>=0.3.4" "apache-tvm-ffi<=0.1.9"
pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

echo "📚 [3/4] Installing Python dependencies & gdown..."
pip install einops huggingface_hub transformers pandas scikit-learn matplotlib seaborn gdown pandas peft

echo "📂 [4/4] Preparing Dataset & Pretrained Weights..."
mkdir -p data
mkdir -p weights

# if [ ! -f "data/turkey_1channel.npz" ]; then
#     echo "Downloading dataset Turkey 1-Channel..."
#     gdown 1HuSEUKz7PEFRAK75Q8RfcPUDiRr8Th_V -O data/turkey_1channel.npz
# else
#     echo "✅ Dataset Turkey 1-Channel sudah ada, melewati proses unduh."
# fi

if [ ! -f "data/turkey_3channel.npz" ]; then
    echo "Downloading dataset Turkey 3-Channel..."
    gdown 1M4W0PFOA7ICC5RqOqAs8EZ9WM0RlFLAU -O data/turkey_3channel.npz
else
    echo "✅ Dataset Turkey 3-Channel sudah ada, melewati proses unduh."
fi

# 2. Download Pretrained Weights
if [ ! -f "weights/pretrained_weights.pth" ]; then
    echo "Downloading Pretrained Weights..."
    gdown 1O_ff_gibel6W0EvC2P9wB3LBsWBB4XoT -O weights/pretrained_weights.pth
else
    echo "✅ Pretrained Weights sudah ada, melewati proses unduh."
fi

echo "================================================="
echo "🎉 Setup Selesai! Environment dan Data Siap."
echo "================================================="