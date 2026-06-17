# 古诗词 LLM 实践：从头训练 · 0.6B 微调 · GRPO 后训练

面向课程实践报告的完整可复现项目。以**中国古典诗词**为特色数据主线，
覆盖大模型训练的三个核心阶段，并在后训练算例中复现 mini 版 **"Aha Moment"**。

## 与作业要求的对应关系

| 作业要求 | 本项目方案 |
| --- | --- |
| 框架三选一 | **minimind**（任务一从头训练）+ **unsloth**（任务二/三，本身也是候选框架之一）。两者组合的原因见下方"方案说明" |
| 特色数据集 | [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（全唐诗/宋词约 30 万首，繁转简后构建语料与指令集）；任务三用 GSM8K 数学题（可验证奖励是产生 Aha Moment 的前提） |
| 算例① 从头训练 | minimind 64M 模型在古诗词语料上预训练（`task1_pretrain_minimind/`） |
| 算例② ≥0.6B 微调 | Qwen3-0.6B + LoRA SFT，微调成"古诗词助手"（`task2_sft_qwen0.6b/`） |
| 算例③ 后训练 | Qwen3-0.6B + **GRPO** 强化学习 on GSM8K，观察 Aha Moment（`task3_grpo_qwen0.6b/`） |
| 文献叙述 | `report/report_outline.md` 已列好问题背景、相关工作与参考文献骨架 |

**方案说明（报告里也建议写明，体现技术判断）**：三个候选框架没有任何
一个能独立覆盖三个算例 —— nanoGPT 无后训练支持、unsloth 不做从头训练、
minimind 官方未发布 0.6B 级权重。因此选 minimind 演示"从零预训练"
（代码极简、单文件可读，最适合在报告中拆解原理），选 unsloth 完成
0.6B 微调与 GRPO（显存省、速度快、官方 GRPO 教程成熟）。

## 仓库结构

```
├── setup_autodl.sh                  # AutoDL 一键环境配置
├── requirements.txt
├── data/
│   ├── download_poetry.sh           # 下载 chinese-poetry 原始数据
│   ├── build_pretrain_corpus.py     # -> data/out/pretrain_guwen.jsonl（任务一语料）
│   └── build_sft_dataset.py         # -> 续写/默写/出处问答 指令集（任务二数据）
├── task1_pretrain_minimind/
│   ├── README.md                    # minimind 从头训练完整操作步骤
│   └── plot_loss_from_log.py        # 从训练日志画 loss 曲线
├── task2_sft_qwen0.6b/
│   ├── train_sft_unsloth.py         # LoRA SFT
│   └── infer_compare.py             # 微调前后同题对比 -> compare.md
├── task3_grpo_qwen0.6b/
│   ├── train_grpo_unsloth.py        # GRPO（规则奖励：正确性+格式）
│   └── eval_gsm8k.py                # 前后准确率对比 + Aha 片段扫描
└── report/
    └── report_outline.md            # 报告骨架 + 文献 + 必截图清单
```

## 完整工作流

### 第一步：本地（VSCode）准备与冒烟验证

数据脚本不依赖 GPU，本地就能把数据集构建好并检查质量：

```bash
pip install opencc-python-reimplemented
bash data/download_poetry.sh          # 约 700MB，慢可挂代理
python data/build_pretrain_corpus.py --max_samples 200000
python data/build_sft_dataset.py
head -n 3 data/out/pretrain_guwen.jsonl   # 肉眼检查
```

### 第二步：push 到 GitHub / Gitee

```bash
git init && git add -A && git commit -m "init: guwen llm practice"
git remote add origin git@github.com:<你的用户名>/guwen-llm-practice.git
git push -u origin main
# 国内访问 GitHub 不稳时，建议同时加 gitee 远程做镜像：
git remote add gitee git@gitee.com:<你的用户名>/guwen-llm-practice.git
git push gitee main
```

> `.gitignore` 已排除 `data/raw`、`data/out` 与权重文件 ——
> 数据在 AutoDL 上重新跑脚本生成即可，仓库保持轻量。

### 第三步：AutoDL 租卡训练

1. 创建实例：GPU 选 **RTX 4090（24G）**，镜像选 PyTorch 2.5.x + Python 3.12 + CUDA 12.x；
2. 拉代码并初始化环境：

```bash
source /etc/network_turbo                  # 学术加速
git clone https://github.com/<你>/guwen-llm-practice.git /root/guwen-llm-practice
cd /root/guwen-llm-practice
bash setup_autodl.sh                       # 装依赖 + 下载 Qwen3-0.6B
bash data/download_poetry.sh
python data/build_pretrain_corpus.py
python data/build_sft_dataset.py
```

3. 三个算例依次跑（**每个先冒烟，再正式跑**）：

```bash
# ---- 任务一：从头训练（详见 task1_pretrain_minimind/README.md）----
# 约 1~2 小时

# ---- 任务二：0.6B LoRA SFT（约 30 分钟）----
python task2_sft_qwen0.6b/train_sft_unsloth.py --max_steps 10   # 冒烟
python task2_sft_qwen0.6b/train_sft_unsloth.py                  # 正式
python task2_sft_qwen0.6b/infer_compare.py                      # 生成对比素材

# ---- 任务三：GRPO 后训练（约 2~3 小时）----
python task3_grpo_qwen0.6b/train_grpo_unsloth.py --max_steps 10 # 冒烟
python task3_grpo_qwen0.6b/train_grpo_unsloth.py                # 正式 500 步
python task3_grpo_qwen0.6b/eval_gsm8k.py --n 200                # 前后准确率+Aha扫描
```

4. 训练曲线（任务二/三）用 tensorboard 看并截图：

```bash
tensorboard --logdir /root/autodl-tmp/outputs --port 6006
# AutoDL 控制台 -> 自定义服务 映射 6006 端口后浏览器打开
```

5. 下载报告素材到本地：loss 曲线 png、`compare.md`、tensorboard 截图、
   `aha_candidates.md`、`summary.json`。AutoDL 文件传输用 JupyterLab
   下载或 `scp` 均可。**跑完记得关机停止计费。**

### 费用估算

4090 约 ¥1.5~2.2/小时。任务一 1~2h + 任务二 0.5h + 任务三 2~3h +
调试余量 ≈ **6~9 卡时，¥15~25** 以内可完成全部算例。

## "Aha Moment" 预期管理（写报告前必读）

DeepSeek-R1 的 Aha Moment（模型自发出现"wait, let me re-check"式反思）
出现在 671B 模型的大规模 RL 中。0.6B + 500 步 GRPO 的小算例里，**合理的
预期**是观察到以下"涌现迹象"，它们都可以作为报告中的 Aha 证据：

1. **reward 曲线台阶式跳变**（先学会 `<reasoning>/<answer>` 格式 → 正确率
   奖励再上台阶）—— tensorboard 截图；
2. **回复长度变化**（`completions/mean_length` 曲线，R1 论文中思维链变长
   与反思行为相伴）；
3. **GSM8K 准确率提升**（`eval_gsm8k.py` 给出训练前后对比数字）；
4. `aha_candidates.md` 中若扫到反思类语句，挑 1~2 条人工确认后截图 ——
   这是最接近字面意义的 Aha Moment，小模型不保证出现，**没有也不影响
   报告结论**（如实写"在小规模下观察到 1/2/3，未稳定出现 4"反而严谨）。
   想提高出现概率，可把基座换成 Qwen3-1.7B（脚本传 `--model_path` 即可，
   显存与时间约翻倍）。

## 常见坑速查

| 问题 | 解决 |
| --- | --- |
| HuggingFace 连不上 | 所有脚本已内置 `HF_ENDPOINT=hf-mirror.com`；确认 `setup_autodl.sh` 已执行 |
| unsloth/vllm 安装冲突 | 按 https://docs.unsloth.ai 选与 torch/cuda 匹配的安装命令；vllm 实在装不上，GRPO 加 `--no_vllm` |
| GRPO 显存 OOM | 降 `--num_generations 4 --batch_size 4`，或 `--max_completion_len 512` |
| trl/unsloth 接口报参数错误 | 两库迭代快，按报错对照官方最新 GRPO notebook 改 1~2 个参数名即可 |
| GSM8K 下载失败 | 已走 hf-mirror；仍失败可手动下载 parquet 后 `load_dataset("parquet", data_files=...)` |
| minimind loss=nan | 学习率减半；检查 jsonl 格式 |
