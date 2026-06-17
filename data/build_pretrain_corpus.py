# -*- coding: utf-8 -*-
"""
任务一数据准备：从 chinese-poetry 构建 MiniMind 预训练语料（古诗词领域）

输入:  data/raw/chinese-poetry/ 下的诗词 JSON（全唐诗/全宋诗/宋词，繁体或简体）
输出:  data/out/pretrain_guwen.jsonl，每行 {"text": "《标题》作者\n诗句..."}
       —— 与 minimind-3 预训练数据 pretrain_t2t(_mini).jsonl 格式完全一致

用法:
    python data/build_pretrain_corpus.py                       # 默认全量
    python data/build_pretrain_corpus.py --limit_files 3       # 本地小样验证
    python data/build_pretrain_corpus.py --max_samples 150000  # 控制语料规模
    # 可选: 混入 minimind 官方通用语料增强语言基础能力
    python data/build_pretrain_corpus.py --mix_jsonl /path/to/pretrain_t2t_mini.jsonl --mix_n 100000
"""
import argparse
import glob
import json
import os
import random

try:
    from opencc import OpenCC  # pip install opencc-python-reimplemented
    CC = OpenCC("t2s")  # 繁体 -> 简体
except ImportError:
    CC = None
    print("[!] 未安装 opencc，将跳过繁简转换：pip install opencc-python-reimplemented")


def t2s(text: str) -> str:
    return CC.convert(text) if CC else text


def collect_json_files(poetry_dir: str, limit_files: int = 0):
    """按优先级收集诗词 JSON 文件（兼容仓库目录结构变化）。"""
    patterns = [
        "**/poet.tang*.json",   # 全唐诗
        "**/poet.song*.json",   # 全宋诗
        "**/ci.song*.json",     # 宋词
    ]
    files, seen = [], set()
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(poetry_dir, pat), recursive=True)):
            if f not in seen:
                seen.add(f)
                files.append(f)
    if not files:
        print("[!] 未按常规命名找到诗词文件，回退为扫描所有含 paragraphs 字段的 JSON ...")
        for f in sorted(glob.glob(os.path.join(poetry_dir, "**/*.json"), recursive=True)):
            files.append(f)
    if limit_files > 0:
        files = files[:limit_files]
    return files


def parse_items(filepath: str):
    """解析单个 JSON 文件，产出 (title, author, body) 三元组。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] 跳过无法解析的文件 {filepath}: {e}")
        return
    if not isinstance(data, list):
        return
    for item in data:
        if not isinstance(item, dict):
            continue
        paras = item.get("paragraphs") or item.get("content") or []
        if not isinstance(paras, list):
            continue
        lines = [str(p).strip() for p in paras if str(p).strip()]
        if not lines:
            continue
        title = str(item.get("title") or item.get("rhythmic") or "").strip()
        author = str(item.get("author") or "").strip()
        body = "\n".join(lines)
        yield title, author, body


def make_text(title: str, author: str, body: str) -> str:
    head = ""
    if title:
        head = f"《{title}》"
    if author:
        head += author
    return t2s(f"{head}\n{body}" if head else body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poetry_dir", default="data/raw/chinese-poetry", help="chinese-poetry 仓库路径")
    ap.add_argument("--out", default="data/out/pretrain_guwen.jsonl", help="输出 jsonl 路径")
    ap.add_argument("--max_samples", type=int, default=0, help="最大样本数，0=不限制")
    ap.add_argument("--min_chars", type=int, default=12, help="过滤过短文本")
    ap.add_argument("--max_chars", type=int, default=480, help="过滤过长文本（minimind 默认截断340 token）")
    ap.add_argument("--limit_files", type=int, default=0, help="只处理前 N 个文件（本地小样验证用）")
    ap.add_argument("--mix_jsonl", default="", help="可选：混入的通用语料 jsonl（如 minimind pretrain_t2t_mini.jsonl）")
    ap.add_argument("--mix_n", type=int, default=0, help="混入通用语料的条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    files = collect_json_files(args.poetry_dir, args.limit_files)
    print(f"[i] 共找到 {len(files)} 个诗词 JSON 文件")

    samples, total_chars, skipped = [], 0, 0
    for fp in files:
        for title, author, body in parse_items(fp):
            text = make_text(title, author, body)
            if not (args.min_chars <= len(text) <= args.max_chars):
                skipped += 1
                continue
            samples.append(text)
        if args.max_samples and len(samples) >= args.max_samples * 2:
            break  # 已远超需要，提前停止解析

    random.shuffle(samples)
    if args.max_samples:
        samples = samples[: args.max_samples]

    # 可选混入通用语料（提升基础语言能力，避免模型只会"之乎者也"）
    mixed = 0
    if args.mix_jsonl and args.mix_n > 0:
        pool = []
        with open(args.mix_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if t:
                    pool.append(t)
                if len(pool) >= args.mix_n * 3:
                    break
        random.shuffle(pool)
        samples += pool[: args.mix_n]
        mixed = min(args.mix_n, len(pool))
        random.shuffle(samples)

    with open(args.out, "w", encoding="utf-8") as f:
        for t in samples:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            total_chars += len(t)

    print(f"[✓] 写出 {len(samples)} 条样本（其中混入通用语料 {mixed} 条），"
          f"约 {total_chars/1e6:.1f}M 字符，过滤 {skipped} 条 -> {args.out}")
    print("[i] 样例预览：")
    for t in samples[:3]:
        print("-" * 40)
        print(t[:120])


if __name__ == "__main__":
    main()
