# -*- coding: utf-8 -*-
"""
任务三：后训练 —— GRPO 强化学习（unsloth + trl），复现 mini 版 "Aha Moment"

在 GSM8K 小学数学题上，用规则奖励（答案正确性 + 输出格式）对 Qwen3-0.6B
做 GRPO 训练。这正是 DeepSeek-R1 论文中产生 "Aha Moment" 的训练范式
（GRPO 出自 DeepSeekMath, arXiv:2402.03300；Aha Moment 见 R1, arXiv:2501.12948）。

报告中重点观察三条曲线（tensorboard）：
  1. reward 总奖励逐步上升（先学会格式 -> 再提升正确率，常呈"台阶式"跳变）
  2. rewards/correctness_reward 正确率奖励
  3. completions/mean_length 回复长度变化（R1 中长度增长伴随反思行为出现）

用法（单卡 4090 约 2~3 小时）:
    python task3_grpo_qwen0.6b/train_grpo_unsloth.py \
        --model_path /root/autodl-tmp/models/Qwen3-0.6B \
        --output_dir /root/autodl-tmp/outputs/task3_grpo
    # 冒烟测试: 加 --max_steps 10
    # vllm 安装失败时: 加 --no_vllm（速度变慢但可跑）

排错提示: unsloth/trl 接口迭代较快，若报参数不匹配，
          以 https://docs.unsloth.ai 的 GRPO 最新示例为准微调本脚本。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import unsloth  # noqa: F401,E402  必须最先导入
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402

import argparse  # noqa: E402
import re  # noqa: E402

from datasets import load_dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

# ------------------------- 提示词与答案抽取 -------------------------
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

PRINT_EVERY = 20  # 每隔多少次奖励计算打印一条样例，便于肉眼捕捉 Aha Moment
_step_counter = {"n": 0}


def strip_think(text: str) -> str:
    """去掉 Qwen3 自带的 <think> 块，再做格式/答案判定。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_xml_answer(text: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    return m.group(1).strip() if m else None


def last_number(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def normalize_num(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def get_pred(completion_text: str):
    text = strip_think(completion_text)
    return normalize_num(extract_xml_answer(text) or last_number(text))


# ------------------------- 奖励函数（规则奖励/RLVR） -------------------------
def correctness_reward(prompts, completions, answer, **kwargs):
    """答案正确 +2.0。这是最核心的可验证奖励。"""
    responses = [c[0]["content"] for c in completions]
    preds = [get_pred(r) for r in responses]
    golds = [normalize_num(a) for a in answer]
    _step_counter["n"] += 1
    if _step_counter["n"] % PRINT_EVERY == 1:
        q = prompts[0][-1]["content"]
        print("\n" + "=" * 60)
        print(f"[样例] Q: {q[:120]}")
        print(f"[样例] 模型输出: {responses[0][:400]}")
        print(f"[样例] 抽取答案: {preds[0]} | 标准答案: {golds[0]}")
    return [2.0 if p is not None and p == g else 0.0 for p, g in zip(preds, golds)]


def int_reward(completions, **kwargs):
    """答案是数字 +0.5（引导模型给出数值答案）。"""
    preds = [get_pred(c[0]["content"]) for c in completions]
    return [0.5 if p is not None and re.fullmatch(r"-?\d+(\.\d+)?", p) else 0.0 for p in preds]


def strict_format_reward(completions, **kwargs):
    """严格符合 <reasoning>...</reasoning>\\n<answer>...</answer> 格式 +0.5。"""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    texts = [strip_think(c[0]["content"]) for c in completions]
    return [0.5 if re.match(pattern, t, flags=re.S) else 0.0 for t in texts]


def soft_format_reward(completions, **kwargs):
    """宽松格式：出现两对标签即 +0.5。"""
    pattern = r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>"
    texts = [strip_think(c[0]["content"]) for c in completions]
    return [0.5 if re.search(pattern, t, flags=re.S) else 0.0 for t in texts]


def xml_count_reward(completions, **kwargs):
    """每出现一个正确标签 +0.125（密集的格式塑形奖励）。"""
    def score(t):
        s = 0.0
        for tag in ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]:
            if t.count(tag) == 1:
                s += 0.125
        return s
    return [score(strip_think(c[0]["content"])) for c in completions]


# ------------------------- 数据 -------------------------
def get_gsm8k(split="train", max_samples=0, dataset_name="openai/gsm8k"):
    data = load_dataset(dataset_name, "main")[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    def proc(x):
        gold = x["answer"].split("####")[-1].strip()
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": gold,
        }
    return data.map(proc)


# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--output_dir", default="/root/autodl-tmp/outputs/task3_grpo")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--max_prompt_len", type=int, default=256)
    ap.add_argument("--max_completion_len", type=int, default=700)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num_generations", type=int, default=8, help="每题采样回答数 G")
    ap.add_argument("--batch_size", type=int, default=8, help="须为 num_generations 的整数倍")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--max_samples", type=int, default=0, help="限制训练题目数，0=全量")
    ap.add_argument("--gpu_mem_util", type=float, default=0.5)
    ap.add_argument("--no_vllm", action="store_true", help="禁用 vllm 加速生成")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------- 1. 模型 ----------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        fast_inference=not args.no_vllm,       # vllm rollout 加速
        max_lora_rank=args.lora_r,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ---------- 2. 数据 ----------
    train_ds = get_gsm8k("train", max_samples=args.max_samples)
    print(f"[i] 训练题目数: {len(train_ds)}")

    # ---------- 3. GRPO 训练 ----------
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=not args.no_vllm,
        learning_rate=args.lr,
        adam_beta1=0.9, adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        logging_steps=1,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "tb"),
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xml_count_reward,
        ],
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.train()

    # ---------- 4. 保存 ----------
    lora_dir = os.path.join(args.output_dir, "grpo_lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"[✓] GRPO LoRA 已保存到: {lora_dir}")
    print("[i] 下一步: 用 eval_gsm8k.py 对比训练前后准确率，并搜寻 Aha Moment 片段")


if __name__ == "__main__":
    main()
