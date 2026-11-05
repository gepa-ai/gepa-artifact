#!/usr/bin/env bash
# prints one run_experiments command for qwen3-8b.  usage: make_command.sh <bench> <prog> <optimizer> [extra args]
B=$1; P=$2; O=$3; shift 3
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$REPO/.venv/bin/python
BI=$(cd "$REPO" && $PY -c "from scripts.experiment_configs import get_benchmarks; print([m.name or m.benchmark.__name__ for m in get_benchmarks()].index('$B'))" 2>/dev/null | tail -1)
OI=$(cd "$REPO" && $PY -c "from scripts.experiment_configs import get_optimizers; print([n for n, _ in get_optimizers()].index('$O'))" 2>/dev/null | tail -1)
LM='{"name": "qwen3-8b", "model": "openai/arbor:qwen/qwen3-8b", "api_key": "API_KEY", "api_base": "http://localhost:{portnum}/v1/", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "launch_kwargs": {"max_context_length": 8192}}'
echo "$PY -m scripts.run_experiments --bm_idx $BI --benchmark_name $B --num_threads 32 --program_idx 0 --prog_name $P --opt_idx $OI --optim_name $O --lm_config '$LM' --seed 0 $*"
