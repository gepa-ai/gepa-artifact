#!/usr/bin/env python3
import argparse
import json
import os
import sys


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _load_lm_config(baseline_config_path, lm_config_json):
    if lm_config_json:
        return json.loads(lm_config_json)
    config = _load_json(baseline_config_path)
    return config.get("lm_config", config)


def _load_prompts(prompts_path, prompts_dir):
    prompts = {}
    if prompts_dir:
        for name in os.listdir(prompts_dir):
            if not name.endswith(".md"):
                continue
            stem = os.path.splitext(name)[0]
            path = os.path.join(prompts_dir, name)
            with open(path, "r") as f:
                content = f.read().strip()
            prompts[stem] = content
    if prompts_path:
        extra = _load_json(prompts_path)
        if not isinstance(extra, dict):
            raise ValueError("prompts_path must be a JSON object mapping predictor names to instructions")
        prompts.update(extra)
    if not prompts:
        raise ValueError("No prompts loaded. Provide --prompts_dir or --prompts_path.")
    return prompts


def _apply_prompts(program, prompts):
    updated = []
    for name, pred in program.named_predictors():
        key = name
        alt = name[:-8] if name.endswith(".predict") else f"{name}.predict"
        if key in prompts:
            new_instruction = prompts[key]
        elif alt in prompts:
            new_instruction = prompts[alt]
        else:
            continue
        pred.signature = pred.signature.with_instructions(new_instruction)
        updated.append(name)
    return updated


def main():
    parser = argparse.ArgumentParser(description="Evaluate custom prompts on the HotpotQA test set.")
    parser.add_argument("--prompts_path", default=None, help="JSON file with predictor->instruction mapping.")
    parser.add_argument(
        "--prompts_dir",
        default=None,
        help="Directory with one prompt per .md file (e.g., summarize1.md).",
    )
    parser.add_argument(
        "--baseline_config_path",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "experiment_runs_data",
            "experiment_runs",
            "seed_0",
            "HotpotQABench_HotpotMultiHop_Baseline_gpt-4o",
            "config.json",
        ),
        help="Baseline config.json to reuse LM settings.",
    )
    parser.add_argument("--lm_config_json", default=None, help="Override LM config with a JSON string.")
    parser.add_argument("--run_dir", required=True, help="Output directory for metric_logs and evaluation_results.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of evaluation threads.")
    parser.add_argument("--log_trace", action="store_true", help="Log trace to metric_logs/test.jsonl.")
    parser.add_argument("--cache_dir", default=None, help="Optional DSPy cache directory.")
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["DSPY_CACHEDIR"] = args.cache_dir
        os.environ["DSP_CACHEDIR"] = args.cache_dir
        os.environ["DSPY_NOTEBOOK_CACHEDIR"] = args.cache_dir
        os.environ["DSP_NOTEBOOK_CACHEDIR"] = args.cache_dir

    lm_config = _load_lm_config(args.baseline_config_path, args.lm_config_json)
    prompts = _load_prompts(args.prompts_path, args.prompts_dir)

    # Import run_experiments.py from scripts/ without requiring a package.
    scripts_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(scripts_dir)
    sys.path.insert(0, project_root)
    sys.path.insert(0, scripts_dir)
    import run_experiments as run_exps  # type: ignore

    import dspy
    from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_metas
    from gepa_artifact.benchmarks.benchmark import EvaluationResult
    from gepa_artifact.utils.metric_logger import MetricWithLogger, CounterWithLock

    lm = run_exps.create_lm(lm_config)
    dspy.configure(lm=lm)

    benchmark_meta = hotpot_metas[0]
    program = benchmark_meta.program[0]
    benchmark = benchmark_meta.benchmark()

    updated = _apply_prompts(program, prompts)
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "prompts_used.json"), "w") as f:
        json.dump({k: prompts.get(k) for k in prompts}, f, indent=2)

    metric_counter = CounterWithLock()
    metric_fn = benchmark_meta.metric
    adapter = dspy.settings.adapter

    with MetricWithLogger(
        metric_fn=metric_fn,
        run_dir=args.run_dir,
        counter_with_lock=metric_counter,
        train_dataset=benchmark.train_set,
        val_dataset=benchmark.val_set,
        test_dataset=benchmark.test_set,
        log_prediction=True,
        log_trace=args.log_trace,
    ) as metric_fn_with_logger:
        evaluator = dspy.Evaluate(
            devset=benchmark.test_set,
            metric=metric_fn_with_logger,
            num_threads=args.num_threads,
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
    eval_results.optimizer = "CustomPromptEval"
    eval_results.optimized_program = program
    run_exps.write_evaluation_result_to_path(eval_results, os.path.join(args.run_dir, "evaluation_results"))

    print(f"Updated predictors: {updated}")
    print(f"Score on test set: {score}")


if __name__ == "__main__":
    main()
