# Baseline inventory

| Baseline | Workload | Entry point | Prompt output |
| --- | --- | --- | --- |
| Original / zero-shot | HotpotQA | `scripts/run_experiments.py` | Baseline run's `metric_logs/prompts/` |
| GEPA | HotpotQA, WebArena, VeruSAGE | `scripts/run_experiments.py`, `webarena/run_gepa_webarena_offline.py`, `scripts/verusage_offline_prompt_opt.py` | Task-specific output path |
| Naive few-shot | HotpotQA | `scripts/naive_fewshot_baseline.py` | `--output_prompts_dir` |
| Teacher-corrected few-shot | HotpotQA | `scripts/teacher_fewshot_baseline.py` | `--output_prompts_dir` |
| Trace-RAG | HotpotQA, VeruSAGE | `baseline_rag/trace_rag.py` | `build-prompt --output_path` |
| TextGrad-style | HotpotQA | `scripts/textgrad_hotpotqa_baseline.py` | `--output_prompts_dir` |
| Full-trace TextGrad | VeruSAGE | `scripts/textgrad_fulltrace_baseline.py` | `--output_dir` |
| Naive few-shot / RAG / TextGrad | WebArena | `webarena/build_webarena_baselines.py` | `webarena/<site>/*_<baseline>.json` |

完整运行命令见 [`HANDOFF.md`](HANDOFF.md#baselines)。HotpotQA 生成的任意
一组 prompt 都可以用 `scripts/eval_custom_prompts.py` 评测。

本项目将 `GEPA` 作为 `GEPA-SGD` 的 baseline；
`GEPA-FullTrainset` 和 `GEPA-Linear` 属于变体或消融。运行方式见
[`HANDOFF.md`](HANDOFF.md#gepa-baseline)。
