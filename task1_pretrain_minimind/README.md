# 算例一 · 从头预训练（MiniMind, ~64M）

使用 [MiniMind](https://github.com/jingyaogong/minimind) 在 chinese-poetry 语料上**从随机初始化**
训练一个约 64M 参数的 Transformer 解码器（hidden 768 / 8 层，RMSNorm + RoPE + SwiGLU，
与 Llama / Qwen 系一致）。MiniMind 训练代码精简、单文件可读，便于完整展示自回归预训练流程。

## 结果

![训练损失曲线](../assets/task1_loss.png)

训练损失自约 6.1 平滑收敛至约 2.7（困惑度 ≈ 15），曲线前陡后平、无发散、无 NaN。
模型习得了五 / 七言的节奏切分与一定的对仗倾向，并掌握语料中“《标题》作者 + 正文”的结构格式；
受限于 64M 容量，其事实性较弱、易混搭名句——体现预训练学到的是语言形式与统计规律，而非可靠知识。

## 模型与数据

| 项 | 配置 |
| --- | --- |
| 架构 | Transformer 解码器（RMSNorm + RoPE + SwiGLU） |
| 参数量 | ~64M（`hidden_size=768`, `num_hidden_layers=8`） |
| 语料 | chinese-poetry 全唐诗 / 全宋诗 / 宋词，约 20 万条样本（≈ 15.8M 字符） |
| 超参 | 2 epoch · batch 32 · lr 5e-4 · `max_seq_len` 340 |

## 复现

### 1. 构建语料（在仓库根目录执行）

```bash
bash data/download_poetry.sh
python data/build_pretrain_corpus.py --max_samples 200000   # → data/out/pretrain_guwen.jsonl
```

> 可选：混入部分通用语料以增强语言基础能力
> `python data/build_pretrain_corpus.py --mix_jsonl <general.jsonl> --mix_n 50000`

### 2. 克隆 MiniMind 并训练

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind.git
cd minimind && pip install -r requirements.txt

cd trainer
python train_pretrain.py \
    --data_path <repo>/data/out/pretrain_guwen.jsonl \
    --epochs 2 --batch_size 32 --learning_rate 5e-4 \
    --max_seq_len 340 --hidden_size 768 --num_hidden_layers 8 \
    2>&1 | tee pretrain.log
```

权重保存为 `minimind/out/pretrain_768.pth`（按 `hidden_size` 命名）。
显存不足时降低 `--batch_size`（如 16 / 8）或 `--max_seq_len`；中断后加 `--from_resume 1` 续训。

### 3. 绘制损失曲线

```bash
python task1_pretrain_minimind/plot_loss_from_log.py --log pretrain.log --out task1_loss.png
```

### 4. 生成测试

```bash
cd minimind/trainer
python eval_llm.py --weight pretrain --hidden_size 768 --num_hidden_layers 8
```

进入交互模式后，预训练模型只会**续写**（未学过对话）。输入诗句开头观察续写效果，例如
`床前明月光，`、`国破山河在，`、`《春日》李白`。对比随机初始化 / 半程 / 训练完三个阶段的续写，
可直观展示“从无序到诗味”的演化。

## 可选：继续 SFT 使其具备对话能力

MiniMind 自带 SFT 脚本，可将本仓库 `data/out/sft_guwen_train.jsonl`（格式兼容）用于训练：

```bash
cd minimind/trainer
python train_full_sft.py --from_weight pretrain \
    --data_path <repo>/data/out/sft_guwen_train.jsonl --max_seq_len 768
# 随后 python eval_llm.py --weight full_sft 即可进行问答式交互
```

## 说明

- 损失出现 NaN：调低学习率，并确认 `pretrain_guwen.jsonl` 每行为 `{"text": "..."}`。
- 显存不足：降低 `--batch_size` 或 `--max_seq_len`。
