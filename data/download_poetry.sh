#!/bin/bash
# 下载古诗词语料库 chinese-poetry（约 1GB，--depth 1 只取最新版本）
# 用法: bash data/download_poetry.sh
# 在 AutoDL 上建议先执行: source /etc/network_turbo   （学术加速，加快 GitHub 访问）
set -e

cd "$(dirname "$0")"
mkdir -p raw
cd raw

if [ -d "chinese-poetry" ]; then
    echo "[i] chinese-poetry 已存在，跳过下载"
else
    echo "[i] 正在克隆 chinese-poetry（全唐诗/全宋诗/宋词等，约1GB）..."
    git clone --depth 1 https://github.com/chinese-poetry/chinese-poetry.git
fi

echo "[i] 完成。目录结构："
ls chinese-poetry | head -20
echo
echo "下一步："
echo "  python data/build_pretrain_corpus.py     # 构建预训练语料（任务一）"
echo "  python data/build_sft_dataset.py         # 构建SFT指令数据（任务一可选 / 任务二）"
