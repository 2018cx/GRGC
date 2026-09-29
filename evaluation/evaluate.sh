#!/usr/bin/env bash
# pairwise multi-dataset eval — calls evaluation_ckpt.py (SGLang).
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# =============================================================================
# CONFIG — Edit this block for your machine (paths are relative to REPO_ROOT).
# Optionally export overrides before running, e.g.  export REPO_ROOT=/abs/path
# =============================================================================

# Repository root containing models/, datasets/, training outputs, tools/
REPO_ROOT="${REPO_ROOT:-$(cd "${HERE}/.." && pwd)}"
export EVAL_REPO_ROOT="$REPO_ROOT"

# Judge HF checkpoint: edit JUDGE_PATH, or export JUDGE=/abs/path (overrides).
JUDGE_PATH="${JUDGE_PATH:-models/Qwen2.5-72B-Instruct}"

# Writes under REPO_ROOT unless you set absolute path starting with /
# Or override entirely: export EVAL_ROOT=/any/path
EVAL_OUTPUT_REL="${EVAL_OUTPUT_REL:-eval_outputs/run}"

# Trained checkpoints: REPO-relative paths to .../global_step_<N>/ (passed as --only-hf-dir).
CHECKPOINT_REL_PATHS=(
)

# HF baselines (MODEL keys under models/). Override: export BASELINE_KEYS_ENV='A B'
BASELINE_KEYS=(Qwen2.5-3B-Instruct Llama-3.2-3B-Instruct)

# Required. Choices: gad, doubao-seqkd, dolly-4out, gad-0429-dolly-normal,
# gad-0430-lmsys-adversarial (alias: gad-0430-lmsys-qwen3b), gpt-seqkd, ...
# Override: export CHECKPOINT_GROUP=...
CHECKPOINT_GROUP="${CHECKPOINT_GROUP:-}"

# Optional: judge tensor parallelism
JUDGE_TP="${JUDGE_TP:-}"

# =============================================================================
# end config
# =============================================================================

if [[ -n "${BASELINE_KEYS_ENV:-}" ]]; then
  # shellcheck disable=SC2206
  BASELINE_KEYS=(${BASELINE_KEYS_ENV})
fi

if ((${#CHECKPOINT_REL_PATHS[@]} == 0)); then
  echo "[!] CHECKPOINT_REL_PATHS is empty: add at least one REPO-relative path to global_step_<N>/ in CONFIG."
  exit 1
fi
if [[ -z "${CHECKPOINT_GROUP}" ]]; then
  echo "[!] CHECKPOINT_GROUP is not set: edit CONFIG, or export CHECKPOINT_GROUP=<name>."
  echo "    Valid choices: \"$PYTHON\" \"$HERE/evaluation_ckpt.py\" --help"
  exit 1
fi

if [[ -n "${JUDGE:-}" ]]; then
  if [[ "$JUDGE" == /* ]]; then
    JUDGE_ABS="$JUDGE"
  else
    JUDGE_ABS="${REPO_ROOT%/}/${JUDGE}"
  fi
elif [[ "${JUDGE_PATH}" == /* ]]; then
  JUDGE_ABS="$JUDGE_PATH"
else
  JUDGE_ABS="${REPO_ROOT%/}/${JUDGE_PATH}"
fi

if [[ -n "${EVAL_ROOT:-}" ]]; then
  :
elif [[ "${EVAL_OUTPUT_REL}" == /* ]]; then
  EVAL_ROOT="$EVAL_OUTPUT_REL"
else
  EVAL_ROOT="${REPO_ROOT%/}/${EVAL_OUTPUT_REL}"
fi

EXTRA=()
if [ -n "$JUDGE_TP" ]; then
  EXTRA+=(--judge-tp "$JUDGE_TP")
fi

STEP_DIRS=()
for rel in "${CHECKPOINT_REL_PATHS[@]}"; do
  rel="${rel#/}"
  STEP_DIRS+=("${REPO_ROOT%/}/${rel}")
done

ONLY_ARGS=()

echo "REPO_ROOT=$REPO_ROOT"
echo "JUDGE=$JUDGE_ABS"
echo "EVAL_ROOT=$EVAL_ROOT"

cd "$REPO_ROOT"
for STEP_DIR in "${STEP_DIRS[@]}"; do
  if [ ! -d "$STEP_DIR" ]; then
    echo "[!] missing checkpoint dir: $STEP_DIR"
    exit 1
  fi
  HF_DIR="${STEP_DIR}/actor/huggingface"
  if [ -d "$HF_DIR" ] && [ -n "$(find "$HF_DIR" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null)" ]; then
    echo "[ok HF] $STEP_DIR"
  else
    echo "[merge] $STEP_DIR"
    mkdir -p "$HF_DIR"
    find "${STEP_DIR}/actor/" -maxdepth 1 -type f ! -name "*.pt" -exec cp -n {} "$HF_DIR/" \; 2>/dev/null || true
    $PYTHON tools/merge_model2hf.py --local_dir "${STEP_DIR}/actor" 2>/dev/null || echo "[!] merge_model2hf failed"
  fi
  ONLY_ARGS+=(--only-hf-dir "$HF_DIR")
done

cd "$HERE"
$PYTHON evaluation_ckpt.py \
  --checkpoint-group "$CHECKPOINT_GROUP" \
  --dataset all \
  --skip-teacher \
  --baseline-only "${BASELINE_KEYS[@]}" \
  "${ONLY_ARGS[@]}" \
  --judge-model "$JUDGE_ABS" \
  "${EXTRA[@]}" \
  --eval-output-base "$EVAL_ROOT" \
  "$@"

echo "Done → $EVAL_ROOT"
