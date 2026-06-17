# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> The parent `C:\project_for_ML\CLAUDE.md` covers the workspace at a glance. This file overrides
> and extends it for work *inside* `guwen-llm-practice/`.

## What this project is

A three-stage LLM training pipeline on classical Chinese poetry, designed to satisfy a course
requirement to demonstrate (1) from-scratch pretrain, (2) ≥0.6B finetune, (3) post-training RL.
The three tasks share one data preparation step but otherwise run independently — they do **not**
chain (task2 does not consume task1's checkpoint). Pick a task to work on; the others are inert.

```
data/                          ─ shared poetry corpus build (local-safe)
  download_poetry.sh           ─ clone chinese-poetry to data/raw/   (~1GB)
  build_pretrain_corpus.py     ─ ─> data/out/pretrain_guwen.jsonl    (task1 input)
  build_sft_dataset.py         ─ ─> data/out/sft_guwen_{train,val}.jsonl (task2 input)
task1_pretrain_minimind/       ─ external minimind repo + 64M model on poetry; this folder only has README + plot script
task2_sft_qwen0.6b/            ─ Qwen3-0.6B + LoRA SFT on the 续写/默写/出处 instruction set
task3_grpo_qwen0.6b/           ─ Qwen3-0.6B + GRPO RL on GSM8K (reward = correctness + format)
report/                        ─ report outline + figure list
setup_autodl.sh                ─ AutoDL bootstrap: deps + Qwen3-0.6B from ModelScope
```

## Run-anywhere vs. AutoDL-only

Hard split — match it when adding new scripts:

| Run-anywhere (laptop / Windows) | AutoDL-only (4090, ~24GB) |
| --- | --- |
| `data/*` (corpus build, opencc 繁→简) | `task1_pretrain_minimind/` (minimind training, ~1–2h) |
| `task2_sft_qwen0.6b/infer_compare.py` (transformers+peft, no unsloth) | `task2_sft_qwen0.6b/train_sft_unsloth.py` (~30min) |
| `task3_grpo_qwen0.6b/eval_gsm8k.py` (transformers+peft, no unsloth) | `task3_grpo_qwen0.6b/train_grpo_unsloth.py` (~2–3h) |

The eval / inference scripts deliberately drop the `unsloth` dependency so they can be re-run on
a fresh box without the heavy stack. Don't fold them back into `unsloth`-using helpers.

Default output dir for everything is `/root/autodl-tmp/outputs/<task>` and default model dir is
`/root/autodl-tmp/models/Qwen3-0.6B`. On a non-AutoDL box you must pass `--model_path` /
`--output_dir` explicitly.

## Smoke vs. full runs

Every training entrypoint takes a small-step flag — use it before any "real" run, even after
trivial code changes, because unsloth/trl break on minor version drift more often than the code:

```bash
python task2_sft_qwen0.6b/train_sft_unsloth.py --max_samples 200      # SFT smoke
python task3_grpo_qwen0.6b/train_grpo_unsloth.py --max_steps 10       # GRPO smoke
python data/build_pretrain_corpus.py --limit_files 3                  # data smoke
```

There is no unit-test command and there shouldn't be — the pipeline is end-to-end ML.

## Non-obvious things that bite

- **`unsloth` must be imported before `transformers` / `trl`.** Both training scripts do
  `import unsloth` at the very top *before* anything else (with `# noqa` to silence linters).
  unsloth monkey-patches transformers/trl on import; reordering will silently disable the
  patching and break gradient checkpointing or vllm rollout.
- **`HF_ENDPOINT=https://hf-mirror.com` is set in three places**: `setup_autodl.sh` (writes to
  bashrc), `os.environ.setdefault(...)` at the top of every training/eval script, and the
  expectation that the user `source /etc/network_turbo` on AutoDL. If a download stalls, check
  whether one of these got skipped — the env var must be set *before* `from datasets import ...`.
- **`build_sft_dataset.py` reuses `build_pretrain_corpus.py`** via `sys.path.insert` (same dir
  import). Don't move either file without updating the import; don't rename `collect_json_files`,
  `parse_items`, or `t2s`.
- **chinese-poetry layout drifts.** `collect_json_files()` first tries known patterns
  (`poet.tang*.json`, `poet.song*.json`, `ci.song*.json`), then falls back to scanning every
  `**/*.json`. If you change the file selection logic, keep the fallback — the upstream repo
  reorganizes occasionally.
- **`data/raw/` is ~1GB and gitignored.** Never commit it. `data/out/` is also gitignored — both
  are rebuilt from scratch on AutoDL. The repo deliberately stays under a few MB.
- **minimind is not vendored.** `task1_pretrain_minimind/` only holds the README and a log-loss
  plotter. Task1 training happens inside a separately-cloned `minimind/trainer/` directory
  (`/root/autodl-tmp/minimind`) which points back at *this* repo's `data/out/pretrain_guwen.jsonl`
  via `--data_path`. Don't try to add a `train_pretrain.py` here.
- **GRPO reward stack is five composable rules**, not one (see `train_grpo_unsloth.py`):
  `correctness +2.0` / `int +0.5` / `strict_format +0.5` / `soft_format +0.5` /
  `xml_count +0.125 per tag`. The mix is intentional — strict alone gives sparse signal, dense
  format shaping (xml_count) lets the model climb the format ladder before correctness kicks in.
  When debugging "reward stuck at 0.5", check which sub-reward is firing.
- **Qwen3 emits `<think>...</think>` blocks even when `enable_thinking=False`.** All scoring code
  (`strip_think()` in train_grpo / eval_gsm8k) and chat-template calls handle this. If you see
  format reward unexpectedly low, check that newly-added scoring code also strips think blocks
  before regex-matching `<reasoning>/<answer>`.
- **`enable_thinking=False` is a Qwen3-only kwarg.** Both scripts wrap `apply_chat_template` in
  `try/except TypeError` so they still work on older transformers. Preserve the fallback when
  editing.
- **GRPO `batch_size` must be a multiple of `num_generations`.** Default is 8/8. If you halve
  one for OOM, halve the other or trl will raise.
- **Qwen3-0.6B is too small for a literal "Aha Moment".** The README is explicit that the
  observable evidence in this scale is (a) reward step-jumps, (b) `completions/mean_length`
  growth, (c) GSM8K accuracy lift — not necessarily "wait, let me re-check"-style text. The Aha
  keyword scanner in `eval_gsm8k.py` is intentionally permissive and its hits need human review;
  don't tighten it to autopass.

## Common commands (project-local cheatsheet)

```bash
# Data (run once, locally is fine)
bash data/download_poetry.sh
python data/build_pretrain_corpus.py --max_samples 200000
python data/build_sft_dataset.py

# Task 1 — minimind, run inside the cloned minimind repo (NOT here)
# See task1_pretrain_minimind/README.md for the exact `cd /root/autodl-tmp/minimind/trainer && python train_pretrain.py ...` line.
python task1_pretrain_minimind/plot_loss_from_log.py \
    --log /root/autodl-tmp/outputs/task1_pretrain.log \
    --out /root/autodl-tmp/outputs/task1_loss.png

# Task 2 — Qwen3-0.6B LoRA SFT
python task2_sft_qwen0.6b/train_sft_unsloth.py                 # full run
python task2_sft_qwen0.6b/infer_compare.py                     # before/after sample table

# Task 3 — GRPO on GSM8K
python task3_grpo_qwen0.6b/train_grpo_unsloth.py               # full run (~500 steps)
python task3_grpo_qwen0.6b/train_grpo_unsloth.py --no_vllm     # if vllm install failed
python task3_grpo_qwen0.6b/eval_gsm8k.py --n 200               # before/after accuracy + Aha scan

# Tensorboard for tasks 2 & 3
tensorboard --logdir /root/autodl-tmp/outputs --port 6006
```

## When changing things

- **Keep numbers consistent across `report/report_outline.md` and the script defaults / README
  tables.** Reported epochs/steps/samples are referenced in the report; bumping a default in a
  script silently invalidates the writeup.
- **The three SFT task templates (`XUXIE_TPL` / `MOXIE_TPL` / `CHUCHU_TPL` in
  `build_sft_dataset.py`) are the contract** the model is finetuned against and what
  `infer_compare.py`'s `EXTRA_PROMPTS` mirror. Adding a new task type means updating both files
  and ideally the report's evaluation section.
- **All output paths are absolute Linux paths in the script defaults** (`/root/autodl-tmp/...`).
  Don't "fix" them to relative paths — that breaks the documented AutoDL workflow. Use CLI
  flags for local override.
