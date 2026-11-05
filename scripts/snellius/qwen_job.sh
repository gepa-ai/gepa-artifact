#!/usr/bin/env bash
#SBATCH --job-name=gepa-qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
# Starts Arbor (vLLM) for Qwen3-8B on one GPU, waits until it answers, then runs the
# run_experiments commands listed in the file given as first argument ({portnum} is filled in).
# Submit from the repo root (logs/ must exist there for the --output file):
#   sbatch -p gpu_h100 -t 05:00:00 scripts/snellius/qwen_job.sh my_commands.txt
# Build the command lines with scripts/snellius/make_command.sh.
set -u
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
mkdir -p logs
export PATH=$HOME/.local/bin:$PATH
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy} WANDB_API_KEY=${WANDB_API_KEY:-dummy} WANDB_MODE=disabled GEPA_USE_WANDB=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
CMDS=$1
nvidia-smi -L
START=$(date +%s)
echo "start $(date) on $(hostname), commit $(git rev-parse --short HEAD), commands: $CMDS"

# arbor writes request/response logs to ./storage, keep them out of the repo
WORK=/scratch-shared/$USER/arbor_$SLURM_JOB_ID
mkdir -p "$WORK"
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
(source "$REPO/gepa_artifact/utils/arbor/.venv/bin/activate" && export PYTHONPATH="$REPO/gepa_artifact/utils/arbor" && cd "$WORK" && python -m arbor.cli serve --arbor-config "$REPO/scripts/snellius/arbor_inference_1gpu.yaml" --port $PORT > "$REPO/logs/arbor-$SLURM_JOB_ID.log" 2>&1) &
ARBOR_PID=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "logs/gpu-$SLURM_JOB_ID.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null; pkill -P $ARBOR_PID 2>/dev/null; kill $ARBOR_PID 2>/dev/null' EXIT

# first request makes arbor start vllm; wait until it returns an answer
READY=""
for i in $(seq 1 90); do
  kill -0 $ARBOR_PID 2>/dev/null || break
  grep -qiE "out of memory|larger than the maximum number of tokens|no available memory for the cache|larger than the available KV cache memory" "$REPO/logs/arbor-$SLURM_JOB_ID.log" && { echo "vllm did not fit on this gpu"; break; }
  if curl -s -m 600 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model": "arbor:qwen/qwen3-8b", "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 64}' | grep -q '"choices"'; then
    READY=1; break
  fi
  sleep 20
done
UP=$(( $(date +%s) - START ))
echo "server ready: ${READY:-no} after $UP s"
[ -n "$READY" ] || { echo "server did not start"; tail -50 "logs/arbor-$SLURM_JOB_ID.log"; exit 1; }

STATUS=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  cmd=${cmd//\{portnum\}/$PORT}
  echo "=== $(date) running: ${cmd:0:160}"
  T0=$(date +%s)
  eval "$cmd" || STATUS=1
  echo "=== took $(( $(date +%s) - T0 )) s, exit so far $STATUS"
done < "$CMDS"

echo "end $(date), total $(( $(date +%s) - START )) s, server start-up $UP s"
cp -r "$WORK/storage/logs" "logs/arbor_requests_$SLURM_JOB_ID" 2>/dev/null
exit $STATUS
