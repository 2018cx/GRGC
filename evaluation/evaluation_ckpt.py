"""Multi-dataset pairwise evaluation (SGLang). Set EVAL_REPO_ROOT to the repository root.

Pass checkpoints via shell --only-hf-dir (see evaluation/evaluate.sh). Optional bulk scan:
`--scan-output-dir` and/or env EVAL_SCAN_OUTPUT_DIRS (repo-relative roots; newline or ':' separated).

--checkpoint-group selects how those roots are traversed (directory layout), not their names.
"""

import json
import os
import re
import glob
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import sglang as sgl
from transformers import AutoTokenizer


TP_SIZE_JUDGE = 8

def _repo_root_from_env_or_file() -> Path:
    for key in ("EVAL_REPO_ROOT", "REPO_ROOT"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _repo_root_from_env_or_file()
JUDGE_MODEL_PATH = os.environ.get(
    "EVAL_JUDGE_DEFAULT",
    str(REPO_ROOT / "models" / "Qwen2.5-72B-Instruct"),
)

BASELINE_MODELS = {
    "Qwen2.5-1.5B-Instruct": REPO_ROOT / "models/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": REPO_ROOT / "models/Qwen2.5-3B-Instruct",
    "Llama-3.2-1B-Instruct": REPO_ROOT / "models/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": REPO_ROOT / "models/Llama-3.2-3B-Instruct",
}

DATASET_CONFIGS = {
    "dolly": {
        "path": REPO_ROOT / "datasets/dolly/dolly.parquet",
        "dir": REPO_ROOT / "datasets/dolly",
    },
    "selfinst": {
        "path": REPO_ROOT / "datasets/self-inst/valid.parquet",
        "dir": REPO_ROOT / "datasets/self-inst",
    },
    "vicuna": {
        "path": REPO_ROOT / "datasets/Vicuna/valid.parquet",
        "dir": REPO_ROOT / "datasets/Vicuna",
    },
    "lmsys": {
        "path": REPO_ROOT / "datasets/lmsys/test.parquet",
        "dir": REPO_ROOT / "datasets/lmsys",
    },
}

EVAL_PROMPT_TEMPLATE = """[User Instruction and Input]
{user_input}

[Assistant 1's Response]
{assistant1_output}

[Assistant 2's Response]
{assistant2_output}

[Evaluation Task]
We would like to request your feedback on the performance of two AI assistants in response to the user instruction and input displayed above.
Please rate the helpfulness, relevance, accuracy, and level of detail of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""


def extract_user_question(content) -> str:
    if hasattr(content, 'tolist'):
        content = content.tolist()
    if isinstance(content, list):
        for msg in content:
            if msg.get('role') == 'user':
                return msg.get('content', '')
    return str(content)


def parse_scores(response: str) -> tuple[Optional[float], Optional[float], str]:
    lines = response.strip().split('\n')
    score1, score2 = None, None
    explanation = ""
    if lines:
        first_line = lines[0].strip()
        match = re.search(r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)', first_line)
        if match:
            try:
                score1 = float(match.group(1))
                score2 = float(match.group(2))
            except ValueError:
                pass
        if len(lines) > 1:
            explanation = '\n'.join(lines[1:]).strip()
        elif not match:
            explanation = response.strip()
    return score1, score2, explanation


def load_dataset(dataset_name: str):

    cfg = DATASET_CONFIGS[dataset_name]
    df = pd.read_parquet(cfg["path"])
    data = df.to_dict('records')
    user_inputs = [extract_user_question(item['content']) for item in data]
    teacher_outputs = [item['teacher_response'] for item in data]
    return user_inputs, teacher_outputs


def load_doubao_datasets(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]
    if cfg["dir"] is None:
        return []
    dataset_dir = cfg["dir"]
    doubao_models = []
    for pq_file in sorted(dataset_dir.glob(f"*doubao*.parquet")):
        df = pd.read_parquet(pq_file)
        resp_cols = [c for c in df.columns if c.endswith('_response') and c != 'teacher_response']
        if len(resp_cols) != 1:
            print(f"  [!] Ambiguous doubao response column {pq_file.name}: {resp_cols}")
            continue
        resp_col = resp_cols[0]
        display_name = f"doubao_{pq_file.stem}"
        responses = df[resp_col].tolist()
        doubao_models.append({"name": display_name, "outputs": responses})
        print(f"  Loaded {pq_file.name}: {len(responses)} rows")
    return doubao_models


def _needs_instruction_format(dir_name: str) -> bool:
    if "lmsys" in dir_name:
        return False
    if "dolly" in dir_name and (
        "noraml" in dir_name
        or "-normal-" in dir_name
        or "-complex-" in dir_name
        or "dolly-normal" in dir_name
        or "normal2" in dir_name
    ):
        return False
    return True


def _list_global_step_dirs_doubao_layout(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    out: list[Path] = []
    for p in output_dir.rglob("*"):
        if not p.is_dir() or not p.name.startswith("global_step_"):
            continue
        try:
            rel = p.relative_to(output_dir)
        except ValueError:
            continue
        if 1 <= len(rel.parts) <= 2:
            out.append(p)
    return sorted(out, key=str)


def _parse_scan_output_dirs_spec(raw: str) -> list[str]:
    out: list[str] = []
    for line in raw.replace("\r", "").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for chunk in line.split(":"):
            c = chunk.strip().lstrip("/")
            if c:
                out.append(c)
    return list(dict.fromkeys(out))

def scan_output_dir_list(cli_dirs: Optional[list[str]]) -> list[str]:
    merged: list[str] = []
    raw = os.environ.get("EVAL_SCAN_OUTPUT_DIRS")
    if raw:
        merged.extend(_parse_scan_output_dirs_spec(raw))
    if cli_dirs:
        merged.extend(str(d).strip().lstrip("/") for d in cli_dirs if str(d).strip())
    return list(dict.fromkeys(merged))


def find_checkpoints_legacy_gad_warm_roots(repo_rel_roots: list[str]) -> list[dict]:
    checkpoints: list[dict] = []
    for rel in repo_rel_roots:
        output_dir = REPO_ROOT / rel
        if not output_dir.is_dir():
            continue
        output_type = output_dir.name.replace("output-", "")
        for ckpt_dir in sorted(glob.glob(str(output_dir / "*/*/global_step_*"))):
            ckpt_path = Path(ckpt_dir)
            hf_dir = ckpt_path / "actor" / "huggingface"
            experiment_name = ckpt_path.parent.name
            group_name = ckpt_path.parent.parent.name
            step = ckpt_path.name
            short_name = (
                f"{output_type}__{group_name}__{experiment_name.replace('gpt5-chat-filtered-', '')}"
                f"_{step.replace('global_step_', 'step_')}"
            )
            path_hint = f"{group_name}/{experiment_name}"
            checkpoints.append({
                "name": short_name,
                "hf_dir": str(hf_dir),
                "needs_conversion": not (
                    hf_dir.exists()
                    and (list(hf_dir.glob("*.safetensors")) or list(hf_dir.glob("*.bin")))
                ),
                "use_instruction": _needs_instruction_format(path_hint),
            })
    return checkpoints


def find_checkpoints_std_output_roots(dir_names: list[str]) -> list[dict]:
    checkpoints: list[dict] = []
    for dir_name in dir_names:
        output_dir = REPO_ROOT / dir_name
        if not output_dir.is_dir():
            continue
        for ckpt_path in _list_global_step_dirs_doubao_layout(output_dir):
            hf_dir = ckpt_path / "actor" / "huggingface"
            step = ckpt_path.name
            if ckpt_path.parent.resolve() == output_dir.resolve():
                exp_short = "root"
            else:
                exp_short = ckpt_path.parent.name.replace("gpt5-chat-filtered-", "")

            if dir_name.startswith("output-gpt/"):
                leaf = dir_name[len("output-gpt/"):].strip("/").split("/", 1)[0]
                dir_short = leaf.replace("output-", "")
                short_name = f"gpt__{dir_short}__{exp_short}_{step.replace('global_step_', 'step_')}"
            else:
                dir_short = dir_name.replace("output-", "")
                short_name = f"{dir_short}__{exp_short}_{step.replace('global_step_', 'step_')}"

            inst_hint = dir_name
            checkpoints.append({
                "name": short_name,
                "hf_dir": str(hf_dir),
                "needs_conversion": not (
                    hf_dir.exists()
                    and (list(hf_dir.glob("*.safetensors")) or list(hf_dir.glob("*.bin")))
                ),
                "use_instruction": _needs_instruction_format(inst_hint),
            })
    return checkpoints


def checkpoint_entries_from_hf_dirs_only(paths: list[str]) -> list[dict]:
    out: list[dict] = []
    for raw in paths:
        hf_dir = Path(raw).expanduser().resolve()
        if hf_dir.name != "huggingface":
            cand = hf_dir / "actor" / "huggingface"
            if cand.is_dir():
                hf_dir = cand
        gs = hf_dir.parent.parent if hf_dir.parent.name == "actor" else None
        if gs is None or not gs.name.startswith("global_step_"):
            short_name = ("__".join(str(p) for p in hf_dir.parts[-6:]))[:120] or "hf_only"
            use_inst = False
        else:
            try:
                parts = gs.relative_to(REPO_ROOT).parts
            except ValueError:
                parts = gs.parts[-4:] if len(gs.parts) >= 4 else gs.parts
            outp = parts[0]
            if len(parts) >= 3:
                exp_short = parts[-2].replace("gpt5-chat-filtered-", "")
            else:
                exp_short = "root"
            dir_short = outp.replace("output-", "") if outp.startswith("output-") else outp
            step = gs.name.replace("global_step_", "step_")
            short_name = f"{dir_short}__{exp_short}_{step}"
            path_hint = f"{outp}/{(parts[-2] if len(parts) >= 3 else '')}"
            use_inst = _needs_instruction_format(path_hint)

        ok_weights = hf_dir.is_dir() and (bool(list(hf_dir.glob("*.safetensors"))) or bool(list(hf_dir.glob("*.bin"))))
        out.append({
            "name": short_name,
            "hf_dir": str(hf_dir),
            "needs_conversion": not ok_weights,
            "use_instruction": use_inst,
        })
    return out


def get_tp_size_for_model(model_name: str, model_path: Optional[str] = None) -> int:
    if model_path:
        cfg_path = Path(model_path) / "config.json"
        if cfg_path.is_file():
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                n = cfg.get("num_attention_heads")
                if isinstance(n, int) and n > 0:
                    for tp in (8, 4, 2, 1):
                        if n % tp == 0:
                            return tp
            except (OSError, json.JSONDecodeError, TypeError):
                pass
    name_lower = model_name.lower()
    if "1.5b" in name_lower and "qwen" in name_lower:
        return 4
    return 8


def generate_responses_with_engine(llm, tokenizer, user_inputs: list[str], batch_size: int = 32, use_instruction_format: bool = False) -> list[str]:
    sampling_params = {"temperature": 0, "top_p": 0.95, "max_new_tokens": 1536}
    all_responses = []
    total_batches = (len(user_inputs) + batch_size - 1) // batch_size

    for i in range(0, len(user_inputs), batch_size):
        batch_inputs = user_inputs[i:i + batch_size]
        batch_idx = i // batch_size + 1
        print(f"    generate batch {batch_idx}/{total_batches}...")

        prompts = []
        for user_input in batch_inputs:
            if use_instruction_format:
                instruction_prompt = (
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{user_input}\n\n### Response:\n"
                )
                messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": instruction_prompt}]
            else:
                messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_input}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted)

        outputs = llm.generate(prompts, sampling_params)
        all_responses.extend([o['text'] for o in outputs])

    return all_responses


def _maybe_random_seed_kw(random_seed: Optional[int]) -> dict:
    if random_seed is None:
        return {}
    return {"random_seed": random_seed}


def generate_responses(model_path: str, model_name: str, user_inputs: list[str], batch_size: int = 32, use_instruction_format: bool = True, random_seed: Optional[int] = None) -> list[str]:
    tp_size = get_tp_size_for_model(model_name, model_path)
    print(f"  load model: {model_path} (TP={tp_size})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        seed_kw = _maybe_random_seed_kw(random_seed)
        llm = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            dtype="bfloat16",
            mem_fraction_static=0.85,
            context_length=4096,
            trust_remote_code=True,
            **seed_kw,
        )
    except Exception as e:
        print(f"  [!] load model failed: {e}")
        return None
    try:
        responses = generate_responses_with_engine(llm, tokenizer, user_inputs, batch_size, use_instruction_format)
    finally:
        llm.shutdown()
    return responses

def run_batch_evaluation(judge_llm, judge_tokenizer, eval_data: list[dict], batch_size: int = 32) -> list[dict]:
    sampling_params = {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 1024}

    eval_prompts = []
    for item in eval_data:
        prompt = EVAL_PROMPT_TEMPLATE.format(
            user_input=item['user_input'],
            assistant1_output=item['model_output'],
            assistant2_output=item['ref_72b_output'],
        )
        messages = [{"role": "system", "content": "You are a fair and objective evaluator."}, {"role": "user", "content": prompt}]
        formatted = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        eval_prompts.append(formatted)

    all_responses = []
    total_batches = (len(eval_prompts) + batch_size - 1) // batch_size
    for i in range(0, len(eval_prompts), batch_size):
        batch = eval_prompts[i:i + batch_size]
        batch_idx = i // batch_size + 1
        print(f"    judge batch {batch_idx}/{total_batches}...")
        outputs = judge_llm.generate(batch, sampling_params)
        all_responses.extend([o['text'] for o in outputs])

    results = []
    for item, response in zip(eval_data, all_responses):
        score1, score2, explanation = parse_scores(response)
        results.append({**item, "score_model": score1, "score_72b": score2, "explanation": explanation, "raw_response": response})
    return results


def compute_stats(results: list[dict]) -> dict:
    valid = [r for r in results if r['score_model'] is not None and r['score_72b'] is not None]
    if not valid:
        return {"error": "No valid scores"}
    avg_model = sum(r['score_model'] for r in valid) / len(valid)
    avg_72b = sum(r['score_72b'] for r in valid) / len(valid)
    wins = sum(1 for r in valid if r['score_model'] > r['score_72b'])
    losses = sum(1 for r in valid if r['score_model'] < r['score_72b'])
    ties = len(valid) - wins - losses
    score_ratios = []
    for r in valid:
        score_sum = r['score_model'] + r['score_72b']
        if score_sum > 0:
            score_ratios.append(r['score_model'] / score_sum)
    avg_score_ratio = sum(score_ratios) / len(score_ratios) if score_ratios else 0
    return {
        "total_samples": len(results), "valid_samples": len(valid),
        "avg_score_model": avg_model, "avg_score_72b": avg_72b,
        "avg_score_ratio": avg_score_ratio,
        "model_wins": wins, "ref_72b_wins": losses, "ties": ties,
        "model_win_rate": wins / len(valid) * 100,
    }


def resolve_judge_config(args) -> tuple[str, int, str, str]:
    judge_path = getattr(args, "judge_model", None) or JUDGE_MODEL_PATH
    judge_path = str(Path(judge_path).resolve())
    if getattr(args, "judge_tp", None) is not None:
        judge_tp = args.judge_tp
    else:
        judge_tp = get_tp_size_for_model("", judge_path)
    try:
        legacy = str(Path(JUDGE_MODEL_PATH).resolve())
        if judge_path == legacy:
            return judge_path, judge_tp, "ref_72b_generations.json", "72b"
    except OSError:
        pass
    safe = re.sub(r"[^\w.-]+", "_", Path(judge_path).name)
    return judge_path, judge_tp, f"ref_judge_{safe}_generations.json", f"judge_{safe}"


_STD_SCAN_GROUPS = frozenset({
    "doubao", "doubao-seqkd", "dolly-4out", "gad-0429-dolly-normal",
    "gad-0430-lmsys-adversarial", "gad-0430-lmsys-qwen3b", "gpt-seqkd",
})


def checkpoint_scan_roots_for_group(checkpoint_group: str, cli_scan: list[str]) -> list[dict]:
    dirs = scan_output_dir_list(cli_scan)
    if checkpoint_group == "gad":
        return find_checkpoints_legacy_gad_warm_roots(dirs)
    if checkpoint_group in _STD_SCAN_GROUPS:
        return find_checkpoints_std_output_roots(dirs)
    return []


def run_single_dataset(dataset_name: str, checkpoint_group: str, eval_output_base: Path, args):
    ds_output = eval_output_base / dataset_name
    ds_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"dataset: {dataset_name} | checkpoint-group: {checkpoint_group}")
    print(f"{'='*80}")

    print(f"\nload {dataset_name}...")
    user_inputs, teacher_outputs = load_dataset(dataset_name)
    print(f"loaded {len(user_inputs)} rows")

    doubao_models = [] if args.skip_teacher else load_doubao_datasets(dataset_name)

    cli_scan = getattr(args, "scan_output_dir", []) or []
    checkpoints = checkpoint_scan_roots_for_group(checkpoint_group, cli_scan)
    print(f"scan: {len(checkpoints)} checkpoint(s)")

    only_dirs = list(getattr(args, "only_hf_dir", None) or [])
    want = {str(Path(p).resolve()) for p in only_dirs}

    if want:
        filtered = [
            c for c in checkpoints
            if str(Path(c["hf_dir"]).resolve()) in want
        ]
        if filtered:
            found = {str(Path(c["hf_dir"]).resolve()) for c in filtered}
            checkpoints = filtered
            print(f"--only-hf-dir: matched {len(checkpoints)} / {len(want)} path(s)")
            missed = want - found
            if missed:
                print("  [!] not in scan results:" + "\n      " + "\n      ".join(sorted(missed)))
        else:
            print("[i] --only-hf-dir only (no scan hits or empty EVAL_SCAN_OUTPUT_DIRS)")
            checkpoints = checkpoint_entries_from_hf_dirs_only(only_dirs)

    if not checkpoints:
        raise SystemExit(
            "[!] no checkpoints: set CHECKPOINT_REL_PATHS in evaluate.sh (→ --only-hf-dir), "
            "and/or export EVAL_SCAN_OUTPUT_DIRS plus optional --scan-output-dir."
        )

    judge_path, judge_tp, ref_json_name, eval_suffix = resolve_judge_config(args)
    ref_file = ds_output / ref_json_name

    print(f"\n--- reference ({dataset_name}) ---")
    print(f"    judge: {judge_path}  TP={judge_tp}")
    if ref_file.exists() and not args.regenerate:
        print(f"reuse {ref_file.name}")
        with open(ref_file, 'r') as f:
            ref_72b_outputs = json.load(f)
    else:
        print("generate reference...")
        tokenizer_ref = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
        ref_seed_kw = _maybe_random_seed_kw(getattr(args, "random_seed", None))
        llm_ref = sgl.Engine(
            model_path=judge_path, tp_size=judge_tp, dtype="bfloat16",
            mem_fraction_static=0.90, context_length=8192, chunked_prefill_size=4096,
            max_running_requests=8, trust_remote_code=True,
            **ref_seed_kw,
        )
        ref_72b_outputs = generate_responses_with_engine(llm_ref, tokenizer_ref, user_inputs, use_instruction_format=args.use_instruction_format)
        llm_ref.shutdown()
        with open(ref_file, 'w', encoding='utf-8') as f:
            json.dump(ref_72b_outputs, f, ensure_ascii=False)

    all_models = []
    checkpoint_only = getattr(args, "checkpoint_only_eval", False)
    baseline_filter = getattr(args, "baseline_only", None)

    def _append_baselines() -> None:
        for name, path in BASELINE_MODELS.items():
            if baseline_filter is not None and name not in baseline_filter:
                continue
            all_models.append({"name": name, "path": str(path), "type": "baseline", "use_instruction": False})

    if not checkpoint_only:
        _append_baselines()

    if not checkpoint_only and not args.skip_teacher:
        all_models.append({"name": f"{dataset_name}-Teacher", "path": None, "type": "teacher", "outputs": teacher_outputs})
        for dm in doubao_models:
            all_models.append({"name": dm["name"], "path": None, "type": "teacher", "outputs": dm["outputs"]})
    for ckpt in checkpoints:
        all_models.append({
            "name": ckpt["name"],
            "path": ckpt["hf_dir"],
            "type": "checkpoint",
            "use_instruction": ckpt.get("use_instruction", True),
        })

    print(f"models to run: {len(all_models)}")

    print(f"\n--- generate ({dataset_name}) ---")
    all_generations = {}
    for idx, model_info in enumerate(all_models):
        model_name = model_info["name"]
        print(f"[{idx+1}/{len(all_models)}] {model_name}")

        if model_info["type"] == "teacher":
            all_generations[model_name] = model_info["outputs"]
            continue

        gen_file = ds_output / f"{model_name}_generations.json"
        if gen_file.exists() and not args.regenerate:
            print("  cached generations, load")
            with open(gen_file, 'r') as f:
                all_generations[model_name] = json.load(f)
            continue

        model_path = model_info["path"]
        if not Path(model_path).exists():
            print(f"  [!] missing path: {model_path}")
            continue

        use_inst = model_info.get("use_instruction", True) and args.use_instruction_format
        rs = getattr(args, "random_seed", None)
        responses = generate_responses(model_path, model_name, user_inputs, use_instruction_format=use_inst, random_seed=rs)
        if responses is None:
            continue
        all_generations[model_name] = responses
        with open(gen_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False)

    print(f"\n--- judge ({dataset_name}) ---")
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
    judge_seed_kw = _maybe_random_seed_kw(getattr(args, "random_seed", None))
    judge_llm = sgl.Engine(
        model_path=judge_path, tp_size=judge_tp, dtype="bfloat16",
        mem_fraction_static=0.90, context_length=8192, chunked_prefill_size=4096,
        max_running_requests=8, trust_remote_code=True,
        **judge_seed_kw,
    )

    all_stats = []
    try:
        for model_name, model_responses in all_generations.items():
            print(f"\njudge: {model_name} vs {eval_suffix}")
            eval_file = ds_output / f"{model_name}_vs_{eval_suffix}_eval_results.jsonl"
            if eval_file.exists() and not args.re_evaluate:
                print("  cached eval, load")
                results = [json.loads(l) for l in open(eval_file) if l.strip()]
                stats = compute_stats(results)
                stats["model_name"] = model_name
                all_stats.append(stats)
                if "error" not in stats:
                    print(f"  score_ratio: {stats['avg_score_ratio']:.4f}, win%: {stats['model_win_rate']:.1f}")
                continue

            eval_data = [{"user_input": ui, "model_output": mo, "ref_72b_output": ro}
                         for ui, mo, ro in zip(user_inputs, model_responses, ref_72b_outputs)]
            results = run_batch_evaluation(judge_llm, judge_tokenizer, eval_data)
            with open(eval_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            stats = compute_stats(results)
            stats["model_name"] = model_name
            all_stats.append(stats)
            if "error" not in stats:
                print(f"  score_ratio: {stats['avg_score_ratio']:.4f}, win%: {stats['model_win_rate']:.1f}")
    finally:
        judge_llm.shutdown()


    summary_file = ds_output / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    all_stats_sorted = sorted(all_stats, key=lambda x: x.get('avg_score_ratio', 0), reverse=True)
    ref_label = "ref" if eval_suffix != "72b" else "72B"
    table_lines = [f"={'='*119}", f"summary: {dataset_name} | {checkpoint_group} vs {ref_label} ({eval_suffix})", f"={'='*119}", "",
                   f"{'model':<65} {'score':<8} {'ref':<8} {'ratio':<10} {'W/L/T':<12} {'win%':<8}", "-"*119]
    for s in all_stats_sorted:
        if "error" not in s:
            name = s['model_name']
            if len(name) > 63:
                name = "..." + name[-60:]
            table_lines.append(f"{name:<65} {s['avg_score_model']:<8.2f} {s['avg_score_72b']:<8.2f} {s['avg_score_ratio']:<10.4f} {s['model_wins']}/{s['ref_72b_wins']}/{s['ties']:<8} {s['model_win_rate']:.1f}%")
    for line in table_lines:
        print(line)
    table_file = ds_output / "summary_table.txt"
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(table_lines) + '\n\n')

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset pairwise evaluation (SGLang)")
    parser.add_argument("--checkpoint-group", required=True,
                        choices=[
                            "gad", "doubao", "doubao-seqkd", "dolly-4out",
                            "gad-0429-dolly-normal", "gad-0430-lmsys-adversarial", "gad-0430-lmsys-qwen3b",
                            "gpt-seqkd",
                        ],
                        help="Layout group for optional EVAL_* scan (gad-0430-lmsys-qwen3b aliases adversarial)")
    parser.add_argument("--dataset", required=True, choices=["dolly", "selfinst", "vicuna", "gpt5", "lmsys-doubao", "all"], help="Dataset or all")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate model outputs")
    parser.add_argument("--re-evaluate", action="store_true", help="Re-run judge only")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip teacher and doubao parquet columns")
    parser.add_argument("--use-instruction-format", action="store_true", default=True)
    parser.add_argument("--no-instruction-format", dest="use_instruction_format", action="store_false")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge HF path (default: EVAL_JUDGE_DEFAULT or Qwen2.5-72B-Instruct)")
    parser.add_argument("--judge-tp", type=int, default=None, help="Judge tensor parallel")
    parser.add_argument("--eval-output-base", type=str, default=None, help="Output root (default: eval_results_<group>_all_datasets_72b under repo)")
    parser.add_argument(
        "--scan-output-dir",
        action="append",
        default=None,
        metavar="REL_DIR",
        help="Optional repo-relative training output roots to scan (repeatable). Also set EVAL_SCAN_OUTPUT_DIRS (newline or ':' separated).",
    )
    parser.add_argument(
        "--only-hf-dir",
        action="append",
        default=None,
        metavar="PATH",
        help="Recommended: absolute .../global_step_*/actor/huggingface (from evaluate.sh CHECKPOINT_REL_PATHS)",
    )
    parser.add_argument(
        "--checkpoint-only-eval",
        action="store_true",
        help="Only listed checkpoints (no baselines, teacher, doubao columns)",
    )
    parser.add_argument(
        "--baseline-only",
        nargs="*",
        default=None,
        metavar="MODEL_KEY",
        help="Subset of BASELINE_MODELS keys; empty list adds no baselines",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="N",
        help="SGLang random_seed (optional reproducibility)",
    )
    args = parser.parse_args()
    if args.only_hf_dir is None:
        args.only_hf_dir = []
    if args.scan_output_dir is None:
        args.scan_output_dir = []
    if args.baseline_only is not None:
        unk = set(args.baseline_only) - set(BASELINE_MODELS)
        if unk:
            parser.error(f"--baseline-only unknown keys {sorted(unk)}; allowed {sorted(BASELINE_MODELS)}")

    if args.eval_output_base:
        eval_output_base = Path(args.eval_output_base)
    else:
        eval_output_base = REPO_ROOT / f"eval_results_{args.checkpoint_group}_all_datasets_72b"
    eval_output_base.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        datasets = ["dolly", "selfinst", "vicuna", "gpt5", "lmsys-doubao"]
    else:
        datasets = [args.dataset]

    jr, jtp, _, jex = resolve_judge_config(args)
    print("=" * 80)
    print(f"checkpoint-group: {args.checkpoint_group}")
    print(f"datasets: {', '.join(datasets)}")
    print(f"judge: {jr} (TP={jtp}, eval_suffix={jex})")
    print(f"output: {eval_output_base}")
    if args.random_seed is not None:
        print(f"random_seed: {args.random_seed}")
    if args.only_hf_dir:
        print(f"only-hf-dir: {args.only_hf_dir}")
    if args.checkpoint_only_eval:
        print("checkpoint-only-eval")
    if args.baseline_only is not None:
        print(f"baseline-only: {args.baseline_only if args.baseline_only else '(none)'}")
    print("=" * 80)

    all_dataset_stats = {}
    for ds in datasets:
        try:
            stats = run_single_dataset(ds, args.checkpoint_group, eval_output_base, args)
            all_dataset_stats[ds] = stats
        except Exception as e:
            print(f"\n[!] {ds} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_dataset_stats) > 1:
        print(f"\n{'='*80}")
        print("cross-dataset summary")
        print(f"{'='*80}")

        model_scores = {}
        for ds, stats_list in all_dataset_stats.items():
            for s in stats_list:
                if "error" not in s:
                    name = s["model_name"]
                    if name not in model_scores:
                        model_scores[name] = {}
                    model_scores[name][ds] = s["avg_score_ratio"]

        ds_list = list(all_dataset_stats.keys())
        header = f"{'model':<55} " + " ".join(f"{ds:<10}" for ds in ds_list) + f" {'mean':<10}"
        print(header)
        print("-" * len(header))

        rows = []
        for name, scores in model_scores.items():
            vals = [scores.get(ds) for ds in ds_list]
            valid_vals = [v for v in vals if v is not None]
            avg = sum(valid_vals) / len(valid_vals) if valid_vals else 0
            display = name if len(name) <= 53 else "..." + name[-50:]
            cols = " ".join(f"{(v or 0):<10.4f}" for v in vals)
            rows.append((avg, f"{display:<55} {cols} {avg:<10.4f}"))

        for _, row in sorted(rows, key=lambda x: x[0], reverse=True):
            print(row)

        cross_file = eval_output_base / "cross_dataset_summary.json"
        with open(cross_file, 'w', encoding='utf-8') as f:
            json.dump(all_dataset_stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"\ncross-dataset summary -> {cross_file}")

    print("\ndone.")


if __name__ == "__main__":
    main()
