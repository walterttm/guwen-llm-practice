# 任务一：用 minimind 从头预训练「古诗词小模型」

本任务使用 [minimind](https://github.com/jingyaogong/minimind) 框架，
在 chinese-poetry 古诗词语料上**从零开始**预训练一个 64M 参数的小模型
（MiniMind-3 默认配置：`hidden_size=768, num_hidden_layers=8`）。

> 设计说明：minimind 的训练代码非常精简（单文件可读），适合在报告中
> 讲清楚预训练的完整流程；64M 模型在单卡 4090 上 1~2 小时即可看到
> loss 从 ~9 降到 ~3 以下，并能续写出有"诗味"的文本。

## 0. 前置：构建语料（本仓库根目录执行）

```bash
bash data/download_poetry.sh                 # 克隆 chinese-poetry 原始数据
python data/build_pretrain_corpus.py         # -> data/out/pretrain_guwen.jsonl
```

可选：混入一部分 minimind 官方通用语料，避免模型只会"之乎者也"：

```bash
# 先从 ModelScope 下载 gongjy/minimind_dataset 里的 pretrain_t2t_mini.jsonl
python data/build_pretrain_corpus.py \
    --mix_jsonl /root/autodl-tmp/datasets/pretrain_t2t_mini.jsonl --mix_n 50000
```

## 1. 克隆 minimind 并安装依赖

```bash
# AutoDL 上先开学术加速，clone GitHub 会快很多
source /etc/network_turbo

cd /root/autodl-tmp
git clone --depth 1 https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt
```

## 2. 启动预训练

minimind 的训练脚本要求在 `trainer/` 目录下执行：

```bash
cd /root/autodl-tmp/minimind/trainer

# 假设本仓库 clone 在 /root/guwen-llm-practice
nohup python train_pretrain.py \
    --data_path /root/guwen-llm-practice/data/out/pretrain_guwen.jsonl \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 340 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    > /root/autodl-tmp/outputs/task1_pretrain.log 2>&1 &

tail -f /root/autodl-tmp/outputs/task1_pretrain.log   # 实时看 loss
```

要点：

- 权重会保存到 `minimind/out/pretrain_768.pth`（按 hidden_size 命名）；
- 中断后可加 `--from_resume 1` 断点续训；
- 显存不足（<24G）就把 `--batch_size` 降到 16/8；
- 日志每行形如 `Epoch:[1/2](100/5000), loss: 3.456, ...`，
  训练结束后用本目录的 `plot_loss_from_log.py` 画 loss 曲线放进报告。

## 3. 画 loss 曲线（报告必备图）

```bash
python /root/guwen-llm-practice/task1_pretrain_minimind/plot_loss_from_log.py \
    --log /root/autodl-tmp/outputs/task1_pretrain.log \
    --out /root/autodl-tmp/outputs/task1_loss.png
```

## 4. 测试模型生成（报告对比素材）

```bash
cd /root/autodl-tmp/minimind/trainer
python eval_llm.py --weight pretrain --hidden_size 768 --num_hidden_layers 8
```

`eval_llm.py` 会进入交互模式。预训练模型只会**续写**（没学过对话），
建议输入诗句开头观察续写效果，例如：

- `《春日》李白`
- `床前明月光，`
- `国破山河在，`

把"训练前（随机权重胡言乱语）/ 训练 0.5 epoch / 训练完成"三个阶段的
续写结果截图对比，是报告里最直观的"从头训练学到了什么"的证据。

## 5.（可选加分）继续做一轮 SFT 让小模型学会对话

minimind 自带 SFT 脚本，可把官方 `sft_mini_512.jsonl` 与本仓库
`data/out/sft_guwen_train.jsonl`（格式完全兼容）合并后训练：

```bash
cd /root/autodl-tmp/minimind/trainer
python train_full_sft.py \
    --from_weight pretrain \
    --data_path /root/guwen-llm-practice/data/out/sft_guwen_train.jsonl \
    --max_seq_len 768
# 然后 python eval_llm.py --weight full_sft 就能像聊天一样问它诗词了
```

## 常见问题

| 现象 | 处理 |
| --- | --- |
| loss 一直 nan | 学习率减半重试；确认数据 jsonl 每行是 `{"text": "..."}` |
| CUDA OOM | 降 `--batch_size`，或 `--max_seq_len 256` |
| clone minimind 超时 | 确认已 `source /etc/network_turbo`，或改用 gitee 镜像 |
| 想要 wandb 曲线 | 训练命令加 `--use_wandb`（需先 `wandb login`） |
