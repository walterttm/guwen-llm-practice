# 实践报告骨架（按此填充即可成稿）

> 使用方法：每节已写好"要点提示 + 该放什么图表"。训练完把素材填进去，
> 删掉所有引用块提示即为成稿。建议正文 8~15 页。

---

# 基于 minimind 与 unsloth 的古诗词大模型训练实践：
# 从头预训练、指令微调到 GRPO 后训练

## 摘要

> 150~250 字：做了什么（三个算例）、用了什么数据（古诗词 + GSM8K）、
> 得到什么结果（loss 收敛、微调前后对比、GRPO 奖励上升与 Aha 迹象）。

## 1. 问题背景与训练目的

> 要点：
> - 通用大模型在**中国古典诗词**这一垂直领域的不足：默写出处易幻觉、
>   续写不合格律、风格现代化 —— 引出"领域语料预训练 + 指令微调"的动机；
> - 小参数模型完整复现训练全流程的教学价值（成本可控、过程可观察）；
> - 后训练阶段：仅靠 SFT 模仿数据，模型不会"自己检查自己"；
>   DeepSeek-R1 证明基于可验证奖励的 RL 能让推理能力涌现（Aha Moment），
>   本实践在 0.6B 小模型上用 GRPO + 数学题复现该范式的缩小版。
> - 明确三个算例各自的目的：①理解预训练数据→loss→生成能力的因果链；
>   ②理解 LoRA 微调如何低成本注入领域能力；③理解 RLVR 如何改变模型行为。

## 2. 相关工作与方法综述

> 每段 3~5 句即可，引文编号对应文末参考文献：
> - **语言模型预训练**：GPT-2 的自回归预训练范式 [1]；本实践的 minimind
>   是其 Transformer Decoder 架构的极简实现（RMSNorm、RoPE、SwiGLU 等
>   与 Llama/Qwen 系一致），可顺带 1 段介绍模型结构。
> - **参数高效微调**：LoRA 低秩适配原理 [2]（冻结主干、训练低秩增量
>   ΔW=BA），以及 unsloth 在其上的工程优化（手写 Triton 核、显存减半）。
> - **指令对齐**：InstructGPT 的 SFT→RM→PPO 三段式 [3]；DPO 把偏好学习
>   简化为分类损失 [4]（说明后训练谱系，引出为何选 GRPO）。
> - **GRPO 与推理涌现**：GRPO 在组内对比估计优势、去掉价值网络 [5]；
>   DeepSeek-R1 用规则奖励（正确性+格式）大规模 RL，训练中自发出现
>   反思式 "Aha Moment" [6]。
> - **基座与数据**：Qwen3 系列与其 thinking/non-thinking 双模式 [7]；
>   chinese-poetry 开源语料 [8]；GSM8K 数学基准 [9]；中文诗歌生成系统
>   可引清华"九歌" [10] 作为领域相关工作。

## 3. 总体方案设计

> - 框架选型论证表（nanoGPT/minimind/unsloth 三者能力对比 → 为什么组合）；
> - 数据流水线图：chinese-poetry 原始 JSON → 繁简转换/清洗 → 预训练语料
>   (`{"text":...}`) 与指令集（续写/默写/出处问答三类）→ 各任务；
> - 三个算例的"模型规模-数据-算力"一览表（64M/0.6B/0.6B，4090 单卡时长）。

## 4. 算例一：从头预训练（minimind, 64M）

> 4.1 设置：模型配置（hidden 768 / 8 层 / ~64M 参数）、语料规模（约 N 万首
>     诗词、X 亿 token）、超参（lr 5e-4、seq_len 340、2 epoch）。
> 4.2 结果：
>   - 【图】loss 曲线（plot_loss_from_log.py 产出）；
>   - 【表/截图】同一开头（如"床前明月光，"）在 随机初始化 / 半程 / 训练完
>     三个阶段的续写对比 —— 展示"从乱码到诗味"的演化；
> 4.3 分析：loss 降到多少、模型学会了什么（五/七言节奏、对仗倾向、
>     《标题》作者 的语料格式），以及局限（事实性差、易混搭名句）。

## 5. 算例二：Qwen3-0.6B 指令微调（unsloth + LoRA）

> 5.1 设置：LoRA r=16、目标模块、数据三类配比（续写/默写/出处问答）、
>     训练 token 量与时长；
> 5.2 结果：
>   - 【图】SFT loss 曲线（tensorboard 截图）；
>   - 【表】infer_compare.py 生成的 compare.md：同题下 基座 vs 微调后；
> 5.3 分析：微调后格式服从性、默写准确性、出处回答的提升；讨论 0.6B
>     仍存在的幻觉（如冷门诗出处张冠李戴），说明 SFT 改变的是"行为风格"
>     而非"知识容量"。

## 6. 算例三：GRPO 后训练与 Aha Moment（unsloth + TRL）

> 6.1 设置：奖励函数设计表（正确性 +2.0 / 整数 +0.5 / 严格格式 +0.5 /
>     宽松格式 +0.5 / XML 标签计数 ±0.125）—— 这是报告的方法亮点，
>     务必解释"为什么规则奖励可验证、不可被 reward hacking"；
>     GRPO 超参（G=8、lr 5e-6、500 步）。
> 6.2 结果：
>   - 【图】reward 总曲线 + correctness 分项曲线（tensorboard）；
>   - 【图】completions/mean_length 长度曲线；
>   - 【表】eval_gsm8k.py 的训练前后准确率（base vs GRPO，N=200）；
>   - 【截图】训练日志中的样例演化 / aha_candidates.md 中的反思片段
>     （若有，注明题目与完整输出）。
> 6.3 分析：对照 R1 论文描述解释观察到的现象 —— 格式奖励先饱和、
>     正确率奖励台阶式上升、长度变化；如实说明小模型小步数下
>     字面 Aha 片段是否稳定出现，并讨论规模的作用。

## 7. 结论与心得

> - 三个算例分别验证了什么；
> - 工程坑总结（HF 镜像、unsloth/trl 版本、显存调参……写 3~5 条，
>   这部分最能体现"实践"含金量）；
> - 展望：更大基座（1.7B）、把 GRPO 用到诗词格律奖励上（平仄/押韵
>   规则同样"可验证"，是一个有趣的延伸设想）。

## 参考文献

> [1] Radford et al. Language Models are Unsupervised Multitask Learners (GPT-2). OpenAI, 2019.
> [2] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
> [3] Ouyang et al. Training language models to follow instructions with human feedback (InstructGPT). arXiv:2203.02155.
> [4] Rafailov et al. Direct Preference Optimization. arXiv:2305.18290.
> [5] Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO). arXiv:2402.03300.
> [6] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
> [7] Qwen Team. Qwen3 Technical Report. arXiv:2505.09388.
> [8] chinese-poetry 项目. https://github.com/chinese-poetry/chinese-poetry
> [9] Cobbe et al. Training Verifiers to Solve Math Word Problems (GSM8K). arXiv:2110.14168.
> [10] Guo et al. Jiuge: A Human-Machine Collaborative Chinese Classical Poetry Generation System. ACL 2019.
> [11] minimind 项目. https://github.com/jingyaogong/minimind
> [12] unsloth 文档. https://docs.unsloth.ai

---

## 附：报告必备素材清单（训练时随手收集）

- [ ] 任务一 loss 曲线 png
- [ ] 任务一 三阶段续写对比截图（eval_llm.py 交互窗口）
- [ ] 任务二 tensorboard loss 截图
- [ ] 任务二 compare.md（前后对比表，直接贴）
- [ ] 任务三 reward / correctness / mean_length 三条 tensorboard 曲线
- [ ] 任务三 summary.json 的前后准确率数字
- [ ] 任务三 aha_candidates.md 反思片段（若有）
- [ ] 各任务的训练命令与时长、GPU 型号、总费用（写进实验设置）
