#!/usr/bin/env bash
# fails when a run has no result; also counts error lines and logged test examples
# (failed examples are scored 0 and not written to test.jsonl, F29)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
R=${GEPA_EXPERIMENT_DIR:-$REPO/experiment_runs_data}/experiment_runs/seed_0/$1
[ -f "$R/evaluation_results/evaluation_result.txt" ] || { echo "no result for $1"; exit 1; }
E=$(cat "$R"/run_log*.txt 2>/dev/null | grep -ciE "error|exception")
P=$(cat "$R"/run_log*.txt 2>/dev/null | grep -c "Failed to parse response")
T=$(wc -l < "$R/metric_logs/test.jsonl" 2>/dev/null || echo 0)
echo "$1: score $(sed -n 2p $R/evaluation_results/evaluation_result.txt | cut -d, -f1), error lines $E, parse failures $P, test examples logged $T"
