# -*- coding: utf-8 -*-
"""
任务二：基于 0.6B 权重的微调 —— Qwen3-0.6B + LoRA（unsloth 框架）

在自建的古诗词指令数据（续写/默写/出处问答）上做 SFT，
得到一个"古诗词助手"。单张 4090(24GB) 约 20-40 分钟。

用法（AutoDL 上，项目根目录执行）:
    python task2_sft_qwen0.6b/train_sft_unsloth.py \
        --model_path /root/autodl-tmp/models/Qwen3-0.6B \
        --data_path data/out/sft_guwen_train.jsonl \
        --output_dir /root/autodl-tmp/outputs/task2_sft

训练曲线: tensorboard --logdir /root/autodl-tmp/outputs/task2_sft --port 6007
排错提示: unsloth/trl 接口迭代较快，若报参数不匹配，
          以 https://docs.unsloth.ai 最新示例为准微调本脚本。
"""
import os

# ---- 国内网络环境：必须在 import unsloth/transformers 之前设置 ----
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import unsloth  # noqa: F401,E402  必须最先导入（它会对 transformers/trl 打补丁）
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

from datasets import Dataset  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from trl import SFTTrainer  # noqa: E402


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
    ap.add_argument("--output_dir", default="/root/autodl-tmp/outputs/task2_sft")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=0, help="0=全量；先用 200 冒烟测试")
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

    # ---------- 3. 训练 ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",          # 结束后统一保存
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "tb"),
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
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
