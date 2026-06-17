# -*- coding: utf-8 -*-
"""
任务三评测：GRPO 前 vs 后 在 GSM8K 测试集上的准确率对比 + "Aha Moment" 片段扫描

只依赖 transformers + peft（不依赖 unsloth / vllm），方便在任何环境复跑。
做三件事：
  1. 用基座 Qwen3-0.6B 和 GRPO 后的 LoRA 模型分别回答 GSM8K test 前 N 题；
  2. 按 <answer>...</answer>（缺失时退化为"最后一个数字"）抽取答案算准确率；
  3. 在 GRPO 模型的输出里扫描反思类关键词（wait / 等等 / re-check 等），
     把命中的完整回答存盘 —— 这些就是报告里的 "Aha Moment" 证据截图素材。

用法（建议训练完后跑，单卡约 20~40 分钟，可用 --n 减题量）:
    python task3_grpo_qwen0.6b/eval_gsm8k.py \
        --model_path /root/autodl-tmp/models/Qwen3-0.6B \
        --lora_path  /root/autodl-tmp/outputs/task3_grpo/grpo_lora \
        --out_dir    /root/autodl-tmp/outputs/task3_grpo/eval \
        --n 200
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 与训练脚本保持一致的系统提示，公平起见两个模型都用
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

# 反思 / 自我纠错类关键词（中英都扫，宁多勿漏，人工再筛）
AHA_PATTERNS = [
    r"\bwait\b", r"\bhmm\b", r"\blet me re-?check\b", r"\blet me think again\b",
    r"\bactually\b", r"\bI made (a|an) (mistake|error)\b", r"\bre-?examine\b",
    r"等等", r"等一下", r"重新检查", r"重新算", r"再想想", r"再检查", r"我算错了",
    r"不对", r"换个思路", r"验证一下",
]
AHA_RE = re.compile("|".join(AHA_PATTERNS), flags=re.I)


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_pred(text: str):
    """优先取 <answer> 标签内最后一个数字；没有标签则取全文最后一个数字。"""
    text = strip_think(text)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    scope = m.group(1) if m else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", scope.replace(",", ""))
    return nums[-1] if nums else None


def extract_gold(answer_field: str):
    """GSM8K 标准答案在 '#### ' 之后。"""
    return answer_field.split("####")[-1].strip().replace(",", "")


def num_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # 旧版 transformers 无 enable_thinking 参数
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def run_eval(model, tokenizer, dataset, tag, max_new_tokens, batch_size, out_dir):
    """返回 (accuracy, aha_hits 列表)，并把全部生成结果写 jsonl 备查。"""
    model.eval()
    records, correct, aha_hits = [], 0, []
    f_out = open(os.path.join(out_dir, f"gen_{tag}.jsonl"), "w", encoding="utf-8")

    for start in range(0, len(dataset), batch_size):
        rows = dataset[start: start + batch_size]
        questions = rows["question"]
        golds = [extract_gold(a) for a in rows["answer"]]
        prompts = [build_prompt(tokenizer, q) for q in questions]

        inputs = tokenizer(prompts, return_tensors="pt",
                           padding=True, truncation=True, max_length=512).to(model.device)
        outs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 评测用贪心，结果可复现
            pad_token_id=tokenizer.eos_token_id,
        )
        for i in range(len(questions)):
            gen = tokenizer.decode(
                outs[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_pred(gen)
            ok = num_equal(pred, golds[i])
            correct += int(ok)
            rec = {"question": questions[i], "gold": golds[i],
                   "pred": pred, "correct": ok, "output": gen}
            records.append(rec)
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if AHA_RE.search(strip_think(gen)):
                aha_hits.append(rec)
        done = min(start + batch_size, len(dataset))
        print(f"[{tag}] {done}/{len(dataset)}  当前准确率 {correct/done:.3f}", flush=True)

    f_out.close()
    return correct / len(dataset), aha_hits


def load_model(model_path, lora_path=None, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto")
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合并后推理更快
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen3-0.6B")
    ap.add_argument("--lora_path", default="/root/autodl-tmp/outputs/task3_grpo/grpo_lora")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/outputs/task3_grpo/eval")
    ap.add_argument("--n", type=int, default=200, help="评测题数（GSM8K test 共 1319 题）")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--skip_base", action="store_true", help="只评 GRPO 模型")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device_ok = torch.cuda.is_available()
    dtype = torch.bfloat16 if device_ok else torch.float32

    print("[*] 加载 GSM8K test ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n and args.n < len(ds):
        ds = ds.select(range(args.n))
    print(f"[*] 评测题数: {len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"  # 批量生成必须左 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    if not args.skip_base:
        print("\n========== 评测基座模型（GRPO 前） ==========")
        base = load_model(args.model_path, None, dtype)
        acc, _ = run_eval(base, tokenizer, ds, "base",
                          args.max_new_tokens, args.batch_size, args.out_dir)
        results["base_acc"] = acc
        del base
        torch.cuda.empty_cache()

    print("\n========== 评测 GRPO 后模型 ==========")
    tuned = load_model(args.model_path, args.lora_path, dtype)
    acc, aha = run_eval(tuned, tokenizer, ds, "grpo",
                        args.max_new_tokens, args.batch_size, args.out_dir)
    results["grpo_acc"] = acc
    results["aha_count"] = len(aha)

    aha_path = os.path.join(args.out_dir, "aha_candidates.md")
    with open(aha_path, "w", encoding="utf-8") as f:
        f.write("# Aha Moment 候选片段（关键词命中，需人工复核）\n\n")
        for i, rec in enumerate(aha, 1):
            f.write(f"## 片段 {i}（答案{'正确' if rec['correct'] else '错误'}）\n\n")
            f.write(f"**题目**: {rec['question']}\n\n")
            f.write(f"**模型输出**:\n\n```\n{rec['output']}\n```\n\n---\n\n")

    summary = os.path.join(args.out_dir, "summary.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n================ 评测结果 ================")
    if "base_acc" in results:
        print(f"  基座准确率 : {results['base_acc']:.3f}")
    print(f"  GRPO 准确率: {results['grpo_acc']:.3f}")
    print(f"  Aha 候选片段: {len(aha)} 条 -> {aha_path}")
    print(f"  汇总: {summary}")
    print("提示: aha_candidates.md 里人工挑 1~2 条最像'反思'的，截图进报告。")


if __name__ == "__main__":
    main()
