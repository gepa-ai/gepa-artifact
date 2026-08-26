# Experiment handoff

This branch preserves the local experiment work built on top of the GEPA
artifact. It includes:

- A simplified `GEPA_SGD` optimizer and the integration needed to run it.
- Separate teacher-LM accounting and optional evaluation skipping.
- HotpotQA baseline and prompt-evaluation scripts.
- Offline trace-RAG, naive few-shot, teacher few-shot, and TextGrad-style
  baselines.
- Offline WebArena baseline builders and generated prompt configurations.
- VeruSAGE prompt-optimization and scalability scripts.

## Setup

Follow the setup steps in [`README.md`](README.md), including cloning the
repository's DSPy and Arbor forks:

```bash
bash setup_gepa_repo.sh
uv sync
```

API credentials are read from environment variables. Do not put credentials in
configuration files. Depending on the experiment, the scripts use
`OPENAI_API_KEY`, `PROXY_KEY`, `ANTHROPIC_API_KEY`, and/or `WANDB_API_KEY`.

## Local-only inputs

Large or generated inputs are intentionally excluded from Git:

- `hotpotQA/`
- `wiki.abstracts.2017.tar.gz`
- `gepa_artifact/benchmarks/hover/wiki.abstracts.2017.jsonl`
- `gepa_artifact/benchmarks/hover/bm25s_retriever/`
- `baseline_rag/*_index*/`
- `baseline_rag/*_output*/`
- `webarena/**/traces/`
- `experiment_runs_data/`, evaluation outputs, metric logs, and DSPy caches

Use `download_hotpotqa.py` to download HotpotQA. Retrieval indexes can be
rebuilt with `baseline_rag/trace_rag.py`; see
[`baseline_rag/README.md`](baseline_rag/README.md).

The `verusage/` directory is also excluded because it is a separate local
checkout plus experiment data. Place that checkout at the repository root when
running the VeruSAGE scripts.

## Main entry points

| Workload | Entry point |
| --- | --- |
| GEPA/GEPA-SGD artifact experiments | `python -m scripts.run_experiments` |
| HotpotQA prompt evaluation | `scripts/eval_custom_prompts.py` |
| HotpotQA naive few-shot | `scripts/naive_fewshot_baseline.py` |
| HotpotQA teacher few-shot | `scripts/teacher_fewshot_baseline.py` |
| HotpotQA TextGrad-style baseline | `scripts/textgrad_hotpotqa_baseline.py` |
| Offline trace-RAG | `baseline_rag/trace_rag.py` |
| VeruSAGE segment optimization | `scripts/verusage_offline_prompt_opt.py` |
| VeruSAGE full-trace optimization | `scripts/verusage_offline_fulltrace_prompt_opt.py` |
| VeruSAGE TextGrad-style baseline | `scripts/textgrad_fulltrace_baseline.py` |
| WebArena offline baselines | `webarena/build_webarena_baselines.py` |

Generated prompt variants are retained under `scripts/prompts*/` and the
site-specific WebArena directories. Raw traces, RAG outputs, and caches are not
retained.
See [`BASELINES.md`](BASELINES.md) for a short baseline inventory.
