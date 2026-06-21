# -*- coding: utf-8 -*-
"""
任务二：基于 0.6B 权重的微调 —— Qwen3-0.6B + LoRA（unsloth 框架）

在自建的古诗词指令数据（续写/默写/出处问答）上做 SFT，
得到一个"古诗词助手"。单张 4090(24GB) 约 20-40 分钟。

用法（AutoDL 上，项目根目录执行）:
    python task2_sft_qwen0.6b/train_sft_unsloth.py \
        --model_path /root/autodl-tmp/models/Qwen3-0.6B \
        --data_path data/out/sft_guwen_train.jsonl \
        --val_path  data/out/sft_guwen_val.jsonl \
        --output_dir /root/autodl-tmp/outputs/task2_sft

训练曲线: tensorboard --logdir /root/autodl-tmp/outputs/task2_sft --port 6007
        曲线包含 train/loss + eval/loss 两条；最佳 ckpt 在 output_dir 下，
        训练结束自动 load_best_model_at_end，再做 LoRA 保存。
排错提示: unsloth/trl 接口迭代较快，若报参数不匹配，
          以 https://docs.unsloth.ai 最新示例为准微调本脚本。
          本脚本已迁移到 trl >= 0.15 的 SFTConfig + processing_class 接口。
"""
import os

# ---- 国内网络环境：必须在 import unsloth/transformers 之前设置 ----
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import unsloth  # noqa: F401,E402  必须最先导入（它会对 transformers/trl 打补丁）
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen3-0.6B",
                    help="本地模型目录（推荐先用 modelscope 下载）或 HF 模型名")
    ap.add_argument("--data_path", default="data/out/sft_guwen_train.jsonl")
    ap.add_argument("--val_path", default="data/out/sft_guwen_val.jsonl",
                    help="验证集路径，文件不存在时自动跳过验证集评测")
    ap.add_argument("--output_dir", default="/root/autodl-tmp/outputs/task2_sft")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="设为正数则覆盖 epochs，按步数训练（冒烟用，如 10）；-1=按 epochs 训练")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=0, help="0=全量；先用 200 冒烟测试")
    ap.add_argument("--eval_steps", type=int, default=100, help="每多少步在验证集上评测一次")
    ap.add_argument("--save_steps", type=int, default=200, help="每多少步保存一次 checkpoint")
    ap.add_argument("--save_merged", action="store_true", help="额外导出合并后的完整权重")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------- 1. 加载模型 + 挂 LoRA ----------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,   # 0.6B 用 16bit LoRA 即可
        dtype=None,           # 自动选择 bf16/fp16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ---------- 2. 数据：conversations -> chat template 文本 ----------
    rows = load_jsonl(args.data_path)
    if args.max_samples:
        rows = rows[: args.max_samples]

    def to_text(conv):
        try:
            # Qwen3 专属参数：关闭思考模式，得到干净的 SFT 模板
            return tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False)

    ds = Dataset.from_list([{"text": to_text(r["conversations"])} for r in rows])
    print(f"[i] 训练样本数: {len(ds)}")
    print("[i] 模板化后的样例:\n" + ds[0]["text"][:400])

    # 验证集（可选）：文件存在就加载，便于训练过程中观察 eval_loss
    eval_ds = None
    if args.val_path and os.path.exists(args.val_path):
        val_rows = load_jsonl(args.val_path)
        eval_ds = Dataset.from_list([{"text": to_text(r["conversations"])} for r in val_rows])
        print(f"[i] 验证样本数: {len(eval_ds)}")
    else:
        print(f"[i] 未找到验证集 {args.val_path}，将不做验证集评测")

    # ---------- 3. 训练 ----------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,            # >0 时覆盖 epochs，按步数训练；-1=按 epochs
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        # 验证集存在时按步评测；存在时也按步保存 checkpoint，方便挑最佳
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if eval_ds is not None else "no",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "tb"),
        seed=args.seed,
        # SFT 专属参数（trl >= 0.15 起放在 SFTConfig 而不是 SFTTrainer 上）
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # 可选：只对 assistant 回复计算 loss（更标准的 SFT 做法）
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("[i] 已启用 response-only loss")
    except Exception as e:
        print(f"[i] 未启用 response-only loss（不影响训练）: {e}")

    stats = trainer.train()
    print(f"[✓] 训练完成: {stats.metrics}")

    # ---------- 4. 保存 ----------
    lora_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"[✓] LoRA 权重已保存到: {lora_dir}")

    if args.save_merged:
        merged_dir = os.path.join(args.output_dir, "merged")
        try:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            print(f"[✓] 合并后的完整权重已保存到: {merged_dir}")
        except Exception as e:
            print(f"[!] 合并保存失败（可只用 LoRA 推理）: {e}")


if __name__ == "__main__":
    main()
