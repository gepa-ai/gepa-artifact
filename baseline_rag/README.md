# Offline Trace-RAG Baseline

This baseline builds a static prompt for a given action by retrieving evidence
from offline VeruSAGE traces. Retrieval is restricted to the same action/module
to keep comparisons fair.

## Build index (offline once)

```bash
python baseline_rag/trace_rag.py build-index \
  --traces_dir verusage/traces \
  --output_dir baseline_rag/index \
  --prompt_dir verusage/agents/prompts
```

## Build a static prompt for an action

```bash
python baseline_rag/trace_rag.py build-prompt \
  --index_dir baseline_rag/index \
  --action instantiate_forall \
  --output_path baseline_rag/output/instantiate_forall.md \
  --prompt_dir verusage/agents/prompts
```

By default, the new prompt is the base prompt plus an offline evidence section.
If you want a teacher LLM to rewrite the prompt, pass LM config options:

```bash
python baseline_rag/trace_rag.py build-prompt \
  --index_dir baseline_rag/index \
  --action instantiate_forall \
  --output_path baseline_rag/output/instantiate_forall.md \
  --lm_config_path path/to/lm_config.json
```

The retrieval query combines:
- Action name + prompt description.
- A short prompt excerpt.
- Common failure keywords from same-action traces.

Evidence is drawn from the same action/module only, and the token/char budget is
controlled by `--top_k`, `--max_doc_chars`, and `--max_total_chars`.

For VeruSAGE traces, you can keep only successful examples:
```bash
python baseline_rag/trace_rag.py build-index \
  --traces_dir verusage/traces \
  --output_dir baseline_rag/verusage_index \
  --prompt_dir verusage/agents/prompts \
  --filter_accepted
```

## HotpotQA example (metric_logs)

```bash
python baseline_rag/trace_rag.py build-index \
  --traces_dir experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_Baseline_gpt-4o/metric_logs/traces \
  --output_dir baseline_rag/hotpot_index \
  --prompt_dir experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_Baseline_gpt-4o/metric_logs/prompts

python baseline_rag/trace_rag.py build-prompt \
  --index_dir baseline_rag/hotpot_index \
  --action summarize1 \
  --output_path baseline_rag/hotpot_output/summarize1.md \
  --prompt_dir experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_Baseline_gpt-4o/metric_logs/prompts
```
