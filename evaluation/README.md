# Evaluation

- **`evaluation_ckpt.py`** — Python entry (multi-dataset, SGLang).
- **`evaluate.sh`** — one shell; **all paths you care about are in its CONFIG block at the top**.

## Edit `evaluate.sh` (CONFIG section)

| Variable | Meaning |
|----------|---------|
| **`REPO_ROOT`** | Repo root (`models/`, `datasets/`, checkpoints). Default = parent of `evaluation/`. You can instead run `export REPO_ROOT=/your/clone` before the script. |
| **`JUDGE_PATH`** | Judge HF directory: relative to **`REPO_ROOT`** (e.g. `models/Qwen2.5-72B-Instruct`) or an absolute path starting with **`/`**. |
| **`EVAL_OUTPUT_REL`** | Evaluation output folder: relative to **`REPO_ROOT`**, or absolute if it starts with **`/`**. Or set **`export EVAL_ROOT=/any/path`** (takes precedence over **`EVAL_OUTPUT_REL`**). |
| **`CHECKPOINT_REL_PATHS`** | **Required (non‑empty)**. One **`REPO_ROOT`-relative** path per checkpoint, ending at **`global_step_<N>/`**. Repo defaults ship **without** insider run names — you must edit for your **`output-*` / `<exp>/global_step_*`**. |
| **`BASELINE_KEYS`** | Baseline keys under **`models/<name>/`**. Override at runtime with **`export BASELINE_KEYS_ENV="Qwen2.5-3B-Instruct"`** (space‑separated). |
| **`CHECKPOINT_GROUP`** | **`evaluation_ckpt.py --checkpoint-group`** (layout only; see `--help`). **`export CHECKPOINT_GROUP=…`** overrides the script. |

Optional env overrides:

- **`export JUDGE=...`** — overrides **`JUDGE_PATH`**; absolute if it starts with **`/`**, else relative to **`REPO_ROOT`**.
- **`export CHECKPOINT_GROUP=...`**
- **`export BASELINE_KEYS_ENV='ModelA ModelB'`**
- **`export JUDGE_TP=8`**
- **`export EVAL_SCAN_OUTPUT_DIRS`** — optional bulk scan: repo-relative training output roots (newline or `:`), same for any checkpoint-group; usually you only need `CHECKPOINT_REL_PATHS` → `--only-hf-dir`.

Extra Python flags:

```bash
bash evaluation/evaluate.sh --regenerate
```

## Run

Fill in **`CHECKPOINT_REL_PATHS`** (at least one) and **`CHECKPOINT_GROUP`** (or the matching **`export`**) in CONFIG first, or the script exits with an error.

```bash
bash evaluation/evaluate.sh
```
