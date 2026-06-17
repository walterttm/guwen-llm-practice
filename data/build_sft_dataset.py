# -*- coding: utf-8 -*-
"""
任务二数据准备：从 chinese-poetry 自动构造"古诗词助手"SFT 指令数据

构造三类指令对（均有可验证的标准答案）：
  1. 续写   —— 给出诗的前半部分，要求补全后半部分
  2. 默写   —— 给出作者+诗名，要求默写全诗
  3. 出处问答 —— 给出一句诗，问出自哪首诗、作者是谁

输出（与 minimind SFT 数据格式一致，同时可直接喂给 Qwen 微调脚本）：
  data/out/sft_guwen_train.jsonl   每行 {"conversations": [{"role":"user",...},{"role":"assistant",...}]}
  data/out/sft_guwen_val.jsonl     验证集（用于训练前后对比生成）

用法:
    python data/build_sft_dataset.py
    python data/build_sft_dataset.py --limit_files 3   # 本地小样验证
"""
import argparse
import json
import os
import random

# 复用预训练脚本中的解析逻辑
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pretrain_corpus import collect_json_files, parse_items, t2s  # noqa: E402

XUXIE_TPL = [
    "下面是{src}的前半部分：\n{first}\n请续写出后半部分。",
    "请补全这首诗。{src}的前半部分是：\n{first}",
]
MOXIE_TPL = [
    "请默写{author}的《{title}》。",
    "请背诵《{title}》，作者是{author}。",
]
CHUCHU_TPL = [
    "诗句“{line}”出自哪首作品？作者是谁？",
    "“{line}”这句诗的出处和作者是什么？",
]


def build_samples(files, args):
    xuxie, moxie, chuchu = [], [], []
    for fp in files:
        for title, author, body in parse_items(fp):
            title, author, body = t2s(title), t2s(author), t2s(body)
            lines = [l for l in body.split("\n") if l]
            n = len(lines)
            if not (2 <= n <= 12) or len(body) > 320:
                continue

            src = f"《{title}》（{author}）" if title and author else (f"《{title}》" if title else "这首诗")

            # 1) 续写：偶数句取一半，奇数句取前 n//2 句
            if n >= 2 and len(xuxie) < args.n_xuxie:
                k = n // 2
                first, rest = "\n".join(lines[:k]), "\n".join(lines[k:])
                q = random.choice(XUXIE_TPL).format(src=src, first=first)
                xuxie.append({"conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": rest},
                ]})

            # 2) 默写：需要标题与作者，且全诗不要太长
            if title and author and n <= 8 and len(moxie) < args.n_moxie:
                q = random.choice(MOXIE_TPL).format(author=author, title=title)
                moxie.append({"conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f"《{title}》（{author}）\n{body}"},
                ]})

            # 3) 出处问答：取首句（去掉过短/过长的句子）
            if title and author and len(chuchu) < args.n_chuchu:
                line = lines[0].rstrip("，。？！；")
                if 5 <= len(line) <= 20:
                    q = random.choice(CHUCHU_TPL).format(line=line)
                    chuchu.append({"conversations": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": f"这句出自{author}的《{title}》。"},
                    ]})

        if (len(xuxie) >= args.n_xuxie and len(moxie) >= args.n_moxie
                and len(chuchu) >= args.n_chuchu):
            break
    return xuxie, moxie, chuchu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poetry_dir", default="data/raw/chinese-poetry")
    ap.add_argument("--out_dir", default="data/out")
    ap.add_argument("--n_xuxie", type=int, default=9000, help="续写样本上限")
    ap.add_argument("--n_moxie", type=int, default=7000, help="默写样本上限")
    ap.add_argument("--n_chuchu", type=int, default=6000, help="出处问答样本上限")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--limit_files", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    files = collect_json_files(args.poetry_dir, args.limit_files)
    print(f"[i] 共找到 {len(files)} 个诗词 JSON 文件")

    xuxie, moxie, chuchu = build_samples(files, args)
    all_samples = xuxie + moxie + chuchu
    random.shuffle(all_samples)

    n_val = max(20, int(len(all_samples) * args.val_ratio)) if len(all_samples) > 100 else max(1, len(all_samples) // 10)
    val, train = all_samples[:n_val], all_samples[n_val:]

    train_path = os.path.join(args.out_dir, "sft_guwen_train.jsonl")
    val_path = os.path.join(args.out_dir, "sft_guwen_val.jsonl")
    for path, rows in [(train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[✓] 续写 {len(xuxie)} / 默写 {len(moxie)} / 出处 {len(chuchu)}")
    print(f"[✓] 训练集 {len(train)} 条 -> {train_path}")
    print(f"[✓] 验证集 {len(val)} 条 -> {val_path}")
    print("[i] 样例预览：")
    for r in train[:2]:
        print("-" * 40)
        print("Q:", r["conversations"][0]["content"][:80].replace("\n", " / "))
        print("A:", r["conversations"][1]["content"][:80].replace("\n", " / "))


if __name__ == "__main__":
    main()
