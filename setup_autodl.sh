#!/bin/bash
# =============================================================
# AutoDL 环境一键初始化（在 AutoDL 实例终端里执行）
#   bash setup_autodl.sh
# 建议镜像: PyTorch 2.5.x / Python 3.12 / CUDA 12.x，GPU: RTX 4090 24G
# =============================================================
set -e

echo "==> [1/4] 开启 AutoDL 学术加速（加速 GitHub / HuggingFace）"
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
else
    echo "    (非 AutoDL 环境，跳过)"
fi

echo "==> [2/4] 配置 HuggingFace 国内镜像"
export HF_ENDPOINT=https://hf-mirror.com
# 写入 bashrc，之后每个新终端自动生效
grep -q "HF_ENDPOINT" ~/.bashrc || echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

echo "==> [3/4] 安装 Python 依赖（unsloth/vllm 体积大，约需 5~10 分钟）"
pip install -r requirements.txt

echo "==> [4/4] 从 ModelScope 下载 Qwen3-0.6B 基座权重（约 1.5GB）"
MODEL_DIR=/root/autodl-tmp/models/Qwen3-0.6B
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "    已存在 $MODEL_DIR，跳过下载"
else
    mkdir -p /root/autodl-tmp/models
    modelscope download --model Qwen/Qwen3-0.6B --local_dir "$MODEL_DIR"
fi

mkdir -p /root/autodl-tmp/outputs

echo ""
echo "============================================================"
echo " 环境就绪。三个算例的入口："
echo "   任务一(从头训练): 见 task1_pretrain_minimind/README.md"
echo "   任务二(0.6B微调): python task2_sft_qwen0.6b/train_sft_unsloth.py"
echo "   任务三(GRPO后训练): python task3_grpo_qwen0.6b/train_grpo_unsloth.py"
echo " 正式跑前建议各加小步数参数冒烟测试一次（详见根目录 README）。"
echo "============================================================"
