# -*- coding: utf-8 -*-
"""
任务二评测：微调前 vs 微调后 同题对比生成（报告素材直接来源）

只依赖 transformers + peft（不依赖 unsloth），在任何环境都能跑。
读取验证集中的真实样本 + 几条固定测试题，分别用 基座Qwen3-0.6B 和
LoRA微调后模型 生成回答，写出 markdown 对比表。

用法:
    python task2_sft_qwen0.6b/infer_compare.py \
        --model_path /root/autodl-tmp/models/Qwen3-0.6B \
        --lora_path  /root/autodl-tmp/outputs/task2_sft/lora \
        --val_path   data/out/sft_guwen_val.jsonl \
        --out        /root/autodl-tmp/outputs/task2_sft/compare.md
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXTRA_PROMPTS = [
    "请默写李白的《静夜思》。",
    "“春眠不觉晓”出自哪首作品？作者是谁？",
    "请写一首描写江南春天的五言绝句。",
]


def build_inputs(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return text


@torch.no_grad()
def generate(model, tokenizer, question, max_new_tokens=256):
    text = build_inputs(tokenizer, question)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # 去掉 Qwen3 可能输出的空思考块
    return resp.replace("<think>", "").replace("</think>", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--lora_path", default="/root/autodl-tmp/outputs/task2_sft/lora")
    ap.add_argument("--val_path", default="data/out/sft_guwen_val.jsonl")
    ap.add_argument("--out", default="/root/autodl-tmp/outputs/task2_sft/compare.md")
    ap.add_argument("--n_val", type=int, default=3, help="从验证集抽取的题目数")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 题目：验证集前 n_val 条 + 固定题
    cases = []  # (question, gold or None)
    if os.path.exists(args.val_path):
        with open(args.val_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(cases) >= args.n_val:
                    break
                conv = json.loads(line)["conversations"]
                cases.append((conv[0]["content"], conv[1]["content"]))
    for q in EXTRA_PROMPTS:
        cases.append((q, None))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype).to(device).eval()

    print("[i] 基座模型生成中 ...")
    base_answers = [generate(model, tokenizer, q) for q, _ in cases]

    print("[i] 加载 LoRA 并生成 ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.lora_path).eval()
    lora_answers = [generate(model, tokenizer, q) for q, _ in cases]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 任务二：Qwen3-0.6B 微调前后对比\n\n")
        for i, ((q, gold), a0, a1) in enumerate(zip(cases, base_answers, lora_answers), 1):
            f.write(f"## 测试 {i}\n\n**问题：**\n\n{q}\n\n")
            if gold:
                f.write(f"**参考答案：**\n\n{gold}\n\n")
            f.write(f"**微调前（基座）：**\n\n{a0}\n\n")
            f.write(f"**微调后（LoRA）：**\n\n{a1}\n\n---\n\n")
    print(f"[✓] 对比结果已写入: {args.out}")
    for (q, _), a0, a1 in zip(cases, base_answers, lora_answers):
        print("=" * 50)
        print("Q :", q.replace("\n", " / ")[:60])
        print("前:", a0.replace("\n", " / ")[:80])
        print("后:", a1.replace("\n", " / ")[:80])


if __name__ == "__main__":
    main()
