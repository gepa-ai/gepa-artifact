#!/usr/bin/env python3
"""
TextGrad-style baseline for optimizing HotpotQA prompts from baseline trace files.

This script:
1) Reads per-example trace JSONs (metric_logs/traces/trace_*.json).
2) Uses an LM to produce "textual gradients" (improvement suggestions) per trace for each prompt.
3) Aggregates gradients and rewrites each prompt into a new instruction.
4) Optionally runs evaluation via scripts/eval_custom_prompts.py-compatible logic.

Typical usage:
  ./.venv/bin/python scripts/textgrad_hotpotqa_baseline.py \
    --baseline_run_dir experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_Baseline_gpt-4o \
    --output_prompts_dir experiment_runs_data/experiment_runs/seed_0/<NEW_RUN>/custom_prompts \
    --run_dir experiment_runs_data/experiment_runs/seed_0/<NEW_RUN> \
    --only_incorrect
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


MULTI_PROMPT_GRADIENT_PROMPT = """You are analyzing an execution trace of a multi-hop question answering system (HotpotQA).

The system is composed of multiple sub-modules, each with its own instruction prompt. We want to improve these prompts so
that the FINAL answer is correct and formatted correctly for exact-match style evaluation.

Here are the current instructions for each sub-module:

<modules>
{modules_block}
</modules>

Below is the full execution trace (all sub-module interactions):
```
{full_trace}
```

Question: {question}
Gold answer: {gold_answer}
Model final answer: {pred_answer}
Was final answer correct? {is_correct}

Task:
Identify issues that could be addressed by improving the instruction prompt(s). Produce concrete, actionable prompt
improvements for EACH module.

Output format (STRICT):
Return ONLY a JSON object inside a ```json code block with EXACTLY these keys:
{prompt_keys_json}

Each value MUST be a list of strings. If you have no suggestions for a module, return an empty list [].
Do not include any other keys or any text outside the JSON code block.
"""


APPLY_GRADIENTS_PROMPT = """I provided an assistant module with the following instructions:

<current_instructions>
{curr_instructions}
</current_instructions>

I ran the module as part of a multi-hop QA system on {num_traces} different inputs and collected suggestions to improve the instructions.

Here are the improvement suggestions (textual gradients) from individual traces:

<gradients>
{aggregated_gradients}
</gradients>

Write improved instructions that incorporate these suggestions.

Guidelines:
1. Make the instructions specific and actionable.
2. Prefer constraints that improve exact-match answers (e.g., output only the answer string when needed).
3. Preserve the module's input/output fields and the required output field name(s).
4. The new instructions must be a complete replacement (not a diff).

Output the complete new instructions between <BEGIN> and <END> tags.
"""


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json_block(text: str) -> dict | None:
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", text)
        for blk in blocks:
            blk = blk.strip()
            if blk.startswith("{") and blk.endswith("}"):
                try:
                    return json.loads(blk)
                except Exception:
                    pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _extract_segment_calls(trace_record: dict) -> list[dict]:
    segments = []
    for key, val in sorted(trace_record.items()):
        if not key.startswith("sub_agent_calling_"):
            continue
        if not isinstance(val, dict):
            continue
        name = val.get("name")
        trace = val.get("trace", [])
        if not name or not trace:
            continue
        segments.append({"key": key, "name": name, "trace": trace})
    return segments


def _format_full_trace(trace_record: dict) -> str:
    segments = _extract_segment_calls(trace_record)
    msgs: list[str] = []
    for seg in segments:
        for msg in seg["trace"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            msgs.append(f"[{seg['name']}] {role}: {content}")
    return "\n\n".join(msgs)


def _prediction_answer(trace_record: dict) -> str:
    pred = trace_record.get("prediction")
    if isinstance(pred, dict) and "answer" in pred:
        return str(pred.get("answer"))
    return str(pred)


def _load_prompts_from_dir(prompts_dir: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for p in sorted(prompts_dir.glob("*.md")):
        prompts[p.stem] = p.read_text(encoding="utf-8").strip()
    if not prompts:
        raise ValueError(f"No .md prompts found in {prompts_dir}")
    return prompts


def _redact_lm_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    for key in ("api_key", "openai_api_key", "anthropic_api_key"):
        if key in cfg and cfg[key]:
            cfg[key] = "<redacted>"
    return cfg


def _set_cache_env(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["DSPY_CACHEDIR"] = cache_dir
    os.environ["DSP_CACHEDIR"] = cache_dir
    os.environ["DSPY_NOTEBOOK_CACHEDIR"] = cache_dir
    os.environ["DSP_NOTEBOOK_CACHEDIR"] = cache_dir


class TextGradHotpotOptimizer:
    def __init__(self, lm, prompt_keys: list[str], base_prompts: dict[str, str]):
        self.lm = lm
        self.prompt_keys = prompt_keys
        self.base_prompts = base_prompts

    def compute_trace_gradients(self, trace_sample: dict, *, max_retries: int = 1) -> dict[str, list[str]]:
        modules_block_lines = []
        for key in self.prompt_keys:
            instr = self.base_prompts.get(key, "")
            modules_block_lines.append(f'<module name="{key}">')
            modules_block_lines.append(instr)
            modules_block_lines.append("</module>")
        modules_block = "\n".join(modules_block_lines)

        prompt = MULTI_PROMPT_GRADIENT_PROMPT.format(
            modules_block=modules_block,
            full_trace=trace_sample["full_content"],
            question=trace_sample.get("question", ""),
            gold_answer=trace_sample.get("gold_answer", ""),
            pred_answer=trace_sample.get("pred_answer", ""),
            is_correct=trace_sample.get("is_correct", ""),
            prompt_keys_json=json.dumps(self.prompt_keys),
        )

        response: str | None = None
        data: dict | None = None
        for attempt in range(max_retries + 1):
            response = self.lm(prompt, max_tokens=2048, temperature=0.0)[0]
            data = _extract_json_block(response or "")
            if isinstance(data, dict):
                break
            if attempt >= max_retries:
                break
            # Retry with a "repair" prompt that converts the previous response into strict JSON.
            prompt = (
                "Your previous output was not valid JSON.\n\n"
                "Convert the following text into a STRICT JSON object inside a ```json code block.\n"
                f"The JSON MUST have EXACTLY these keys: {json.dumps(self.prompt_keys)}\n"
                "Each value MUST be a list of strings.\n"
                "Do not include any other keys or any text outside the JSON code block.\n\n"
                "<text>\n"
                f"{response}\n"
                "</text>\n"
            )
            continue

        if not isinstance(data, dict):
            # Graceful fallback: return empty gradients for this trace.
            return {k: [] for k in self.prompt_keys}

        gradients: dict[str, list[str]] = {}
        for key in self.prompt_keys:
            val = data.get(key, [])
            if val is None:
                gradients[key] = []
                continue
            if isinstance(val, str):
                gradients[key] = [val.strip()] if val.strip() else []
                continue
            if isinstance(val, list):
                out: list[str] = []
                for item in val:
                    if item is None:
                        continue
                    s = str(item).strip()
                    if s:
                        out.append(s)
                gradients[key] = out
                continue
            gradients[key] = [str(val).strip()] if str(val).strip() else []

        return gradients

    def apply_gradients(self, prompt_key: str, gradients_by_trace: list[dict]) -> str:
        curr_instructions = self.base_prompts[prompt_key]
        aggregated = ""
        for i, item in enumerate(gradients_by_trace, 1):
            label = item.get("label", f"Trace {i}")
            grads = item.get("gradients", [])
            if not grads:
                continue
            aggregated += f"=== {label} ===\n"
            for g in grads:
                aggregated += f"- {g}\n"
            aggregated += "\n"

        if not aggregated.strip():
            return curr_instructions

        prompt = APPLY_GRADIENTS_PROMPT.format(
            curr_instructions=curr_instructions,
            num_traces=len(gradients_by_trace),
            aggregated_gradients=aggregated.strip(),
        )
        response = self.lm(prompt, max_tokens=1024, temperature=0.0)[0]
        match = re.search(r"<BEGIN>([\\s\\S]*?)<END>", response or "")
        if match:
            return match.group(1).strip()
        return (response or "").strip()


def _load_trace_samples(traces_dir: Path, only_incorrect: bool, max_traces: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    trace_paths = sorted(traces_dir.glob("*.json"))
    if not trace_paths:
        raise ValueError(f"No trace JSON files found in {traces_dir}")

    samples: list[dict] = []
    for p in trace_paths:
        rec = _load_json(p)
        is_correct = bool(rec.get("metric_output"))
        if only_incorrect and is_correct:
            continue
        samples.append(
            {
                "trace_path": str(p),
                "example_id": rec.get("example_id"),
                "question": rec.get("question", ""),
                "gold_answer": rec.get("gold_answer", ""),
                "pred_answer": _prediction_answer(rec),
                "is_correct": is_correct,
                "full_content": _format_full_trace(rec),
            }
        )

    if max_traces is not None and len(samples) > max_traces:
        rng.shuffle(samples)
        samples = samples[:max_traces]
        samples = sorted(samples, key=lambda x: x["trace_path"])

    return samples


def _load_lm_config(baseline_config_path: Path, lm_config_json: str | None) -> dict:
    if lm_config_json:
        return json.loads(lm_config_json)
    cfg = _load_json(baseline_config_path)
    return cfg.get("lm_config", cfg)


def _evaluate_with_custom_prompts(
    prompts_dir: Path,
    baseline_config_path: Path,
    lm_config_json: str | None,
    run_dir: Path,
    num_threads: int,
    log_trace: bool,
    cache_dir: str | None,
) -> float:
    if cache_dir:
        os.environ["DSPY_CACHEDIR"] = cache_dir
        os.environ["DSP_CACHEDIR"] = cache_dir
        os.environ["DSPY_NOTEBOOK_CACHEDIR"] = cache_dir
        os.environ["DSP_NOTEBOOK_CACHEDIR"] = cache_dir

    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(scripts_dir))

    import eval_custom_prompts as custom_eval  # type: ignore
    import run_experiments as run_exps  # type: ignore

    import dspy
    from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_metas
    from gepa_artifact.benchmarks.benchmark import EvaluationResult
    from gepa_artifact.utils.metric_logger import MetricWithLogger, CounterWithLock

    lm_config = custom_eval._load_lm_config(str(baseline_config_path), lm_config_json)
    prompts = custom_eval._load_prompts(None, str(prompts_dir))

    lm = run_exps.create_lm(lm_config)
    dspy.configure(lm=lm)

    benchmark_meta = hotpot_metas[0]
    program = benchmark_meta.program[0]
    benchmark = benchmark_meta.benchmark()

    updated = custom_eval._apply_prompts(program, prompts)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "prompts_used.json").open("w", encoding="utf-8") as f:
        json.dump({k: prompts.get(k) for k in prompts}, f, indent=2)

    metric_counter = CounterWithLock()
    metric_fn = benchmark_meta.metric
    adapter = dspy.settings.adapter

    with MetricWithLogger(
        metric_fn=metric_fn,
        run_dir=str(run_dir),
        counter_with_lock=metric_counter,
        train_dataset=benchmark.train_set,
        val_dataset=benchmark.val_set,
        test_dataset=benchmark.test_set,
        log_prediction=True,
        log_trace=log_trace,
    ) as metric_fn_with_logger:
        evaluator = dspy.Evaluate(
            devset=benchmark.test_set,
            metric=metric_fn_with_logger,
            num_threads=num_threads,
            display_progress=True,
            max_errors=len(benchmark.test_set) * 10,
            provide_traceback=True,
        )
        dspy.configure(lm=lm, adapter=adapter)
        score = evaluator(program)

    eval_results = EvaluationResult(
        benchmark=benchmark_meta.name or benchmark.__class__.__name__,
        program=getattr(program, "_name", program.__class__.__name__),
    )
    eval_results.score = score
    eval_results.cost, eval_results.input_tokens, eval_results.output_tokens = run_exps.calculate_stats(lm)
    eval_results.optimizer = "TextGradHotpotBaseline"
    eval_results.optimized_program = program
    run_exps.write_evaluation_result_to_path(eval_results, str(run_dir / "evaluation_results"))

    print(f"Updated predictors: {updated}")
    print(f"Score on test set: {score}")
    return float(score)


def main():
    parser = argparse.ArgumentParser(description="TextGrad-style prompt optimization baseline for HotpotQA.")
    parser.add_argument(
        "--baseline_run_dir",
        default=str(
            Path(__file__).resolve().parent.parent
            / "experiment_runs_data"
            / "experiment_runs"
            / "seed_0"
            / "HotpotQABench_HotpotMultiHop_Baseline_gpt-4o"
        ),
        help="Baseline run directory containing metric_logs/traces and metric_logs/prompts.",
    )
    parser.add_argument(
        "--traces_dir",
        default=None,
        help="Directory with per-example trace JSON files (defaults to baseline_run_dir/metric_logs/traces).",
    )
    parser.add_argument(
        "--base_prompts_dir",
        default=None,
        help="Directory with baseline prompts .md files (defaults to baseline_run_dir/metric_logs/prompts).",
    )
    parser.add_argument(
        "--baseline_config_path",
        default=None,
        help="Baseline config.json to reuse LM settings (defaults to baseline_run_dir/config.json).",
    )
    parser.add_argument("--lm_config_json", default=None, help="Override LM config with a JSON string.")
    parser.add_argument(
        "--output_prompts_dir",
        required=True,
        help="Directory to write optimized prompts (one .md per predictor).",
    )
    parser.add_argument(
        "--prompt_keys",
        default=None,
        help="Comma-separated prompt keys to optimize (defaults to all .md stems in base_prompts_dir).",
    )
    parser.add_argument("--only_incorrect", action="store_true", help="Use only incorrect traces for gradients.")
    parser.add_argument("--max_traces", type=int, default=None, help="Max number of traces to use (after filtering).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for trace sampling.")
    parser.add_argument(
        "--max_traces_per_prompt_in_apply",
        type=int,
        default=None,
        help="If set, subsample this many traces per prompt when applying gradients (helps keep context small).",
    )
    parser.add_argument("--run_dir", default=None, help="If set, run evaluation and save results here.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of evaluation threads.")
    parser.add_argument("--log_trace", action="store_true", help="Log trace during eval to metric_logs/test.jsonl.")
    parser.add_argument("--cache_dir", default=None, help="Optional DSPy cache directory for eval.")
    args = parser.parse_args()

    baseline_run_dir = Path(args.baseline_run_dir)
    traces_dir = Path(args.traces_dir) if args.traces_dir else baseline_run_dir / "metric_logs" / "traces"
    base_prompts_dir = (
        Path(args.base_prompts_dir) if args.base_prompts_dir else baseline_run_dir / "metric_logs" / "prompts"
    )
    baseline_config_path = (
        Path(args.baseline_config_path) if args.baseline_config_path else baseline_run_dir / "config.json"
    )

    output_prompts_dir = Path(args.output_prompts_dir)
    output_prompts_dir.mkdir(parents=True, exist_ok=True)

    if not args.cache_dir:
        args.cache_dir = str(output_prompts_dir / ".dspy_cache")
    _set_cache_env(args.cache_dir)

    base_prompts = _load_prompts_from_dir(base_prompts_dir)
    if args.prompt_keys:
        prompt_keys = [x.strip() for x in args.prompt_keys.split(",") if x.strip()]
    else:
        prompt_keys = sorted(base_prompts.keys())

    missing = [k for k in prompt_keys if k not in base_prompts]
    if missing:
        raise ValueError(f"Missing prompts in base_prompts_dir {base_prompts_dir}: {missing}")

    # Setup LM for TextGrad steps (reuse baseline LM config by default).
    import dspy
    from scripts.run_experiments import create_lm

    lm_config = _load_lm_config(baseline_config_path, args.lm_config_json)
    lm = create_lm(lm_config)
    dspy.configure(lm=lm)

    # Load trace samples.
    trace_samples = _load_trace_samples(
        traces_dir=traces_dir,
        only_incorrect=args.only_incorrect,
        max_traces=args.max_traces,
        seed=args.seed,
    )
    if not trace_samples:
        raise ValueError("No trace samples found after filtering.")
    print(f"[info] Using {len(trace_samples)} traces from {traces_dir}")

    # Compute gradients per trace for all prompts in a single LM call per trace.
    optimizer = TextGradHotpotOptimizer(lm=lm, prompt_keys=prompt_keys, base_prompts=base_prompts)

    gradients_per_prompt: dict[str, list[dict]] = {k: [] for k in prompt_keys}
    per_trace_records: list[dict] = []
    for idx, sample in enumerate(trace_samples, 1):
        label = f"trace={Path(sample['trace_path']).name} example_id={sample.get('example_id')} correct={sample.get('is_correct')}"
        print(f"[gradients] {idx}/{len(trace_samples)} {label}")
        grads = optimizer.compute_trace_gradients(sample)
        per_trace_records.append({"label": label, "trace_path": sample["trace_path"], "gradients": grads})
        for k in prompt_keys:
            gradients_per_prompt[k].append({"label": label, "gradients": grads.get(k, [])})

    # Rewrite prompts.
    rng = random.Random(args.seed)
    written: list[str] = []
    for k in prompt_keys:
        items = gradients_per_prompt[k]
        if args.max_traces_per_prompt_in_apply is not None and len(items) > args.max_traces_per_prompt_in_apply:
            items = list(items)
            rng.shuffle(items)
            items = items[: args.max_traces_per_prompt_in_apply]
            items = sorted(items, key=lambda x: x["label"])
        print(f"[apply] {k}: applying gradients from {len(items)} traces")
        new_instr = optimizer.apply_gradients(k, items)
        out_path = output_prompts_dir / f"{k}.md"
        out_path.write_text(new_instr.strip() + "\n", encoding="utf-8")
        written.append(str(out_path))

    # Write metadata + raw gradients for inspection.
    meta = {
        "baseline_run_dir": str(baseline_run_dir),
        "traces_dir": str(traces_dir),
        "base_prompts_dir": str(base_prompts_dir),
        "baseline_config_path": str(baseline_config_path),
        "lm_config": _redact_lm_config(lm_config),
        "prompt_keys": prompt_keys,
        "only_incorrect": bool(args.only_incorrect),
        "max_traces": args.max_traces,
        "seed": args.seed,
        "num_traces_used": len(trace_samples),
        "max_traces_per_prompt_in_apply": args.max_traces_per_prompt_in_apply,
        "outputs": {"prompts": written},
    }
    (output_prompts_dir / "textgrad_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (output_prompts_dir / "textgrad_trace_gradients.json").write_text(
        json.dumps(per_trace_records, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[done] Wrote optimized prompts to {output_prompts_dir}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        _evaluate_with_custom_prompts(
            prompts_dir=output_prompts_dir,
            baseline_config_path=baseline_config_path,
            lm_config_json=args.lm_config_json,
            run_dir=run_dir,
            num_threads=args.num_threads,
            log_trace=args.log_trace,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
