#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys


DEFAULT_BASELINE_RUN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "experiment_runs_data",
    "experiment_runs",
    "seed_0",
    "HotpotQABench_HotpotMultiHop_Baseline_gpt-4o",
)


def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _set_cache_env(cache_dir):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["DSPY_CACHEDIR"] = cache_dir
    os.environ["DSP_CACHEDIR"] = cache_dir
    os.environ["DSPY_NOTEBOOK_CACHEDIR"] = cache_dir
    os.environ["DSP_NOTEBOOK_CACHEDIR"] = cache_dir


def _extract_sub_agent_calls(record):
    calls = {}
    for key, value in record.items():
        if not key.startswith("sub_agent_calling_"):
            continue
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        trace = value.get("trace")
        if name and trace:
            calls[name] = trace
    return calls


def _get_trace_content(trace, role):
    for msg in trace:
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


def _format_prompt(system_instruction, examples):
    lines = [system_instruction.strip(), ""]
    lines.append("Teacher-corrected few-shot examples (from incorrect training traces):")
    for idx, (user_msg, assistant_msg) in enumerate(examples, 1):
        lines.append(f"Example {idx}:")
        lines.append("Input:")
        lines.append(user_msg.strip())
        lines.append("Output:")
        lines.append(assistant_msg.strip())
        lines.append("----")
    return "\n".join(lines).rstrip() + "\n"


def _prompt_stem(name):
    return name[:-8] if name.endswith(".predict") else name


def _metric_is_correct(metric_output):
    if isinstance(metric_output, dict):
        for key in ("correct", "score", "exact_match", "em"):
            if key in metric_output:
                return bool(metric_output[key])
        if len(metric_output) == 1:
            return bool(next(iter(metric_output.values())))
        return False
    return bool(metric_output)


def _format_gold_answer(gold_answer):
    if isinstance(gold_answer, list):
        return "; ".join(str(item) for item in gold_answer)
    return str(gold_answer)


def _generate_teacher_output(teacher_lm, user_msg, gold_answer):
    system_prompt = (
        "You are given a question and intermediate summaries from a multi-hop QA system. "
        "Produce a correct reasoning and answer. Use the exact format:\n"
        "Reasoning: <brief reasoning>\n"
        "Answer: <final answer>\n"
        "The final answer must match the provided gold answer exactly."
    )
    user_content = f"{user_msg.strip()}\nGold answer: {gold_answer}"
    outputs = teacher_lm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )
    if not outputs:
        raise RuntimeError("Teacher LM returned no outputs.")
    return outputs[0].strip()


def build_teacher_fewshot_prompts(
    records,
    num_shots,
    seed,
    shuffle,
    teacher_lm,
    teacher_target_predictor,
    gold_answer_field,
):
    predictor_names = None
    candidates = []
    for record in records:
        if _metric_is_correct(record.get("metric_output")):
            continue
        calls = _extract_sub_agent_calls(record)
        if not calls:
            continue
        if predictor_names is None:
            predictor_names = sorted(calls.keys())
        if all(name in calls for name in predictor_names):
            candidates.append((record, calls))

    if predictor_names is None:
        raise ValueError("No sub_agent_calling traces found in training traces.")
    if not candidates:
        raise ValueError("No incorrect records contain all required sub_agent_calling traces.")
    if num_shots > len(candidates):
        raise ValueError(f"Requested {num_shots} shots, but only {len(candidates)} incorrect examples available.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    selected = candidates[:num_shots] if num_shots > 0 else candidates
    example_ids = [rec.get("example_id") for rec, _ in selected]

    teacher_outputs = []
    for record, calls in selected:
        if teacher_target_predictor not in calls:
            raise ValueError(f"Missing teacher target predictor {teacher_target_predictor}.")
        trace = calls[teacher_target_predictor]
        user_msg = _get_trace_content(trace, "user")
        gold_answer = _format_gold_answer(record.get(gold_answer_field, ""))
        teacher_outputs.append(_generate_teacher_output(teacher_lm, user_msg, gold_answer))

    prompts = {}
    for name in predictor_names:
        system_instruction = None
        examples = []
        for idx, (_, calls) in enumerate(selected):
            trace = calls[name]
            if system_instruction is None:
                system_instruction = _get_trace_content(trace, "system")
            user_msg = _get_trace_content(trace, "user")
            if name == teacher_target_predictor:
                assistant_msg = teacher_outputs[idx]
            else:
                assistant_msg = _get_trace_content(trace, "assistant")
            if not user_msg or not assistant_msg:
                continue
            examples.append((user_msg, assistant_msg))

        if not system_instruction:
            raise ValueError(f"Missing system instruction for {name}.")
        if not examples:
            raise ValueError(f"No usable examples found for {name}.")

        prompts[name] = _format_prompt(system_instruction, examples)

    return prompts, predictor_names, example_ids


def write_prompts(prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, content in prompts.items():
        filename = f"{_prompt_stem(name)}.md"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def write_metadata(output_dir, metadata):
    path = os.path.join(output_dir, "teacher_fewshot_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def evaluate_prompts(
    prompts_dir,
    baseline_config_path,
    lm_config_json,
    run_dir,
    num_threads,
    log_trace,
    cache_dir,
):
    _set_cache_env(cache_dir)

    scripts_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(scripts_dir)
    sys.path.insert(0, project_root)
    sys.path.insert(0, scripts_dir)

    import eval_custom_prompts as custom_eval  # type: ignore
    import run_experiments as run_exps  # type: ignore

    import dspy
    from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_metas
    from gepa_artifact.benchmarks.benchmark import EvaluationResult
    from gepa_artifact.utils.metric_logger import MetricWithLogger, CounterWithLock

    lm_config = custom_eval._load_lm_config(baseline_config_path, lm_config_json)
    prompts = custom_eval._load_prompts(None, prompts_dir)

    lm = run_exps.create_lm(lm_config)
    dspy.configure(lm=lm)

    benchmark_meta = hotpot_metas[0]
    program = benchmark_meta.program[0]
    benchmark = benchmark_meta.benchmark()

    updated = custom_eval._apply_prompts(program, prompts)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "prompts_used.json"), "w", encoding="utf-8") as f:
        json.dump({k: prompts.get(k) for k in prompts}, f, indent=2)

    metric_counter = CounterWithLock()
    metric_fn = benchmark_meta.metric
    adapter = dspy.settings.adapter

    with MetricWithLogger(
        metric_fn=metric_fn,
        run_dir=run_dir,
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
    eval_results.optimizer = "TeacherCorrectedFewShotBaseline"
    eval_results.optimized_program = program
    run_exps.write_evaluation_result_to_path(eval_results, os.path.join(run_dir, "evaluation_results"))

    print(f"Updated predictors: {updated}")
    print(f"Score on test set: {score}")


def main():
    parser = argparse.ArgumentParser(
        description="Build teacher-corrected few-shot prompts from incorrect training traces and optionally evaluate."
    )
    parser.add_argument(
        "--baseline_run_dir",
        default=DEFAULT_BASELINE_RUN_DIR,
        help="Baseline run directory with metric_logs/train_traces.jsonl.",
    )
    parser.add_argument(
        "--train_traces_path",
        default=None,
        help="Path to train_traces.jsonl (defaults to baseline_run_dir/metric_logs/train_traces.jsonl).",
    )
    parser.add_argument(
        "--output_prompts_dir",
        required=True,
        help="Directory to write teacher-corrected few-shot prompts (one .md per predictor).",
    )
    parser.add_argument("--num_shots", type=int, default=3, help="Number of few-shot examples to include.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling examples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training traces before sampling shots.")
    parser.add_argument(
        "--baseline_config_path",
        default=None,
        help="Baseline config.json to reuse LM settings (defaults to baseline_run_dir/config.json).",
    )
    parser.add_argument("--lm_config_json", default=None, help="Override LM config with a JSON string.")
    parser.add_argument(
        "--teacher_lm_config_json",
        default=None,
        help="JSON string for teacher LM config (defaults to lm_config_json/baseline).",
    )
    parser.add_argument(
        "--teacher_target_predictor",
        default="final_answer.predict",
        help="Predictor name whose outputs are replaced with teacher-corrected reasoning and answer.",
    )
    parser.add_argument(
        "--gold_answer_field",
        default="gold_answer",
        help="Field name for gold answers in the training trace records.",
    )
    parser.add_argument("--run_dir", default=None, help="If set, run evaluation and save results here.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of evaluation threads.")
    parser.add_argument("--log_trace", action="store_true", help="Log trace to metric_logs/test.jsonl.")
    parser.add_argument("--cache_dir", default=None, help="Optional DSPy cache directory.")
    args = parser.parse_args()

    train_traces_path = args.train_traces_path
    if not train_traces_path:
        train_traces_path = os.path.join(args.baseline_run_dir, "metric_logs", "train_traces.jsonl")

    if not args.cache_dir:
        args.cache_dir = os.path.join(args.output_prompts_dir, ".dspy_cache")
    _set_cache_env(args.cache_dir)

    scripts_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(scripts_dir)
    sys.path.insert(0, project_root)
    sys.path.insert(0, scripts_dir)

    import eval_custom_prompts as custom_eval  # type: ignore
    import run_experiments as run_exps  # type: ignore

    lm_config = custom_eval._load_lm_config(
        args.baseline_config_path or os.path.join(args.baseline_run_dir, "config.json"),
        args.lm_config_json,
    )
    teacher_lm_config = lm_config
    if args.teacher_lm_config_json:
        teacher_lm_config = custom_eval._resolve_env_references(
            json.loads(args.teacher_lm_config_json)
        )

    teacher_lm = run_exps.create_lm(teacher_lm_config)

    records = _load_jsonl(train_traces_path)
    prompts, predictor_names, example_ids = build_teacher_fewshot_prompts(
        records,
        args.num_shots,
        args.seed,
        args.shuffle,
        teacher_lm,
        args.teacher_target_predictor,
        args.gold_answer_field,
    )
    write_prompts(prompts, args.output_prompts_dir)

    metadata = {
        "baseline_run_dir": args.baseline_run_dir,
        "train_traces_path": train_traces_path,
        "num_shots": args.num_shots,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "predictors": predictor_names,
        "example_ids": example_ids,
        "teacher_target_predictor": args.teacher_target_predictor,
        "gold_answer_field": args.gold_answer_field,
    }
    write_metadata(args.output_prompts_dir, metadata)

    if args.run_dir:
        baseline_config_path = args.baseline_config_path or os.path.join(
            args.baseline_run_dir, "config.json"
        )
        evaluate_prompts(
            prompts_dir=args.output_prompts_dir,
            baseline_config_path=baseline_config_path,
            lm_config_json=args.lm_config_json,
            run_dir=args.run_dir,
            num_threads=args.num_threads,
            log_trace=args.log_trace,
            cache_dir=args.cache_dir,
        )
    else:
        print(f"Wrote teacher-corrected few-shot prompts to {args.output_prompts_dir}")


if __name__ == "__main__":
    main()
