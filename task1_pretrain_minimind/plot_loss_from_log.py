# -*- coding: utf-8 -*-
r"""
从 minimind 训练日志中提取 loss 并绘制曲线（任务一报告配图）。

minimind 日志行形如:
    Epoch:[1/2](100/5000), loss: 3.456, logits_loss: 3.4, aux_loss: 0.05, lr: ..., epoch_time: ...
正则 r"\bloss: ([\d.]+)" 只会命中独立的 "loss:"，不会误抓 logits_loss / aux_loss。

用法:
    python task1_pretrain_minimind/plot_loss_from_log.py \
        --log /root/autodl-tmp/outputs/task1_pretrain.log \
        --out /root/autodl-tmp/outputs/task1_loss.png
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境
import matplotlib.pyplot as plt

LOSS_RE = re.compile(r"\bloss: ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="train_pretrain.py 的输出日志")
    ap.add_argument("--out", default="task1_loss.png")
    ap.add_argument("--smooth", type=int, default=20, help="滑动平均窗口，0=不平滑")
    args = ap.parse_args()

    losses = []
    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if m:
                losses.append(float(m.group(1)))

    if not losses:
        raise SystemExit("[!] 日志中没有匹配到 'loss: x.xxx'，确认传入的是训练日志？")

    print(f"[*] 共解析到 {len(losses)} 个 loss 记录点，"
          f"首个 {losses[0]:.3f} -> 最后 {losses[-1]:.3f}")

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(losses, alpha=0.35, label="raw loss")
    if args.smooth > 1 and len(losses) > args.smooth:
        win = args.smooth
        sm = [sum(losses[max(0, i - win + 1): i + 1]) / len(losses[max(0, i - win + 1): i + 1])
              for i in range(len(losses))]
        plt.plot(sm, color="crimson", label=f"moving avg (w={win})")
    plt.xlabel("logging step")
    plt.ylabel("train loss")
    plt.title("MiniMind pretraining on classical-poetry corpus")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"[✓] 曲线已保存: {args.out}")


if __name__ == "__main__":
    main()
