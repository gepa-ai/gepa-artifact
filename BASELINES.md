# Baseline inventory

## TextGrad-style

- HotpotQA entry point: `scripts/textgrad_hotpotqa_baseline.py`
- VeruSAGE full-trace entry point: `scripts/textgrad_fulltrace_baseline.py`
- Generated HotpotQA prompts and run metadata are written under
  `experiment_runs_data/experiment_runs/`.

## Naive few-shot

- Entry point: `scripts/naive_fewshot_baseline.py`
- Retained prompt artifacts:
  `scripts/prompts_naive_fewshot/naive_fewshot_metadata.json`
- Evaluation runs and `prompts_used.json` are generated under
  `experiment_runs_data/experiment_runs/`.

## Retrieval-augmented generation

- Documentation: `baseline_rag/README.md`
- Entry point: `baseline_rag/trace_rag.py`
- BM25 indexes and generated prompt outputs are excluded from Git.
