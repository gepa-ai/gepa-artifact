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

## HotpotQA

### 1. Base agent 在哪里

- 主 agent：[`gepa_artifact/benchmarks/hotpotQA/hotpot_program.py`](gepa_artifact/benchmarks/hotpotQA/hotpot_program.py)
  中的 `HotpotMultiHop`。
- 流程：两轮 ColBERT 检索，加上 `summarize1`、`create_query_hop2`、
  `summarize2`、`final_answer` 四个 DSPy 模块。
- Benchmark、metric 和反馈函数注册在
  [`gepa_artifact/benchmarks/hotpotQA/__init__.py`](gepa_artifact/benchmarks/hotpotQA/__init__.py)。
- ColBERT 地址写在 `hotpot_program.py` 中，运行前需确认服务可访问。

### 2. 怎么跑实验

在 [`scripts/experiment_configs.py`](scripts/experiment_configs.py) 中配置
`LM_CONFIGS` 和 optimizer，然后运行：

```bash
bash setup_gepa_repo.sh
uv sync
export PROXY_KEY=...
export WANDB_API_KEY=...
uv run python -m scripts.generate_launch_commands > launch_commands
bash launch_commands
```

结果保存在 `experiment_runs_data/experiment_runs/seed_<seed>/`。需要复用
Baseline cache 的实验必须先跑完 Baseline。

#### 测评自定义 prompt

`prompts_dir` 下的文件名需对应四个模块，例如
`summarize1.md`、`create_query_hop2.md`、`summarize2.md` 和
`final_answer.md`。

```bash
cd /path/to/gepa-artifact

export OPENAI_API_KEY=...
LM_CONFIG='{"name":"gpt-4o","model":"openai/gpt-4o","api_key":"env:OPENAI_API_KEY","temperature":0.0}'
RUN_NAME=HotpotQABench_HotpotMultiHop_CustomPrompts_gpt-4o

.venv/bin/python scripts/eval_custom_prompts.py \
  --prompts_dir scripts/prompts_revised \
  --run_dir "experiment_runs_data/experiment_runs/seed_0/${RUN_NAME}" \
  --lm_config_json "$LM_CONFIG" \
  --num_threads 1
```

也可以用 `--baseline_config_path` 读取已有运行的 LM 配置，但不要提交
包含密钥的 `config.json`。每次评测应使用新的 `RUN_NAME`，避免日志追加
到旧结果中。评测结果、日志和实际使用的 prompt 会写入 `run_dir`。

### 3. 数据集在哪里

- 加载代码：[`gepa_artifact/benchmarks/hotpotQA/hotpot_data.py`](gepa_artifact/benchmarks/hotpotQA/hotpot_data.py)。
- 数据源：Hugging Face `hotpot_qa/fullwiki`；运行时由 `load_dataset(...)`
  下载或读取 Hugging Face cache。
- 如需在 repo 下保存副本，运行 `uv run python download_hotpotqa.py`，
  文件会写入 `hotpotQA/`。该目录被 Git 忽略，且当前 benchmark 不直接从中读取。

#### Training/test set 怎么划分

自定义 split 不会保存成单独的数据文件，而是在
[`gepa_artifact/benchmarks/benchmark.py`](gepa_artifact/benchmarks/benchmark.py)
中动态生成：

- 只读取 HotpotQA 官方 `train` split，共 90,447 条；官方
  `validation` 和 `test` **没有被使用**。
- 前 40% 作为 test，中间 40% 作为 validation，最后 20% 作为
  training。
- 再以固定随机种子 `1` 抽样为：training 150 条、validation 300 条、
  test 300 条。

#### Trace 在哪里

Baseline 的完整 trace 位于：

```text
experiment_runs_data/experiment_runs/seed_0/
  HotpotQABench_HotpotMultiHop_Baseline_gpt-4o/metric_logs/
```

- Training trace：`train_traces.jsonl`，同一批数据也拆成了
  `traces/trace_*.json`。
- Test trace：`test_traces.jsonl`。
- 当前提交包含 150 条 training trace 和 299 条 test trace。
- 普通运行日志：`train.jsonl`、`val.jsonl`、`test.jsonl`；是否包含完整
  trace 取决于运行脚本是否开启 `log_trace`。

上述 Baseline trace、逐条 trace 和对应 prompt 已纳入 Git；其他
`experiment_runs_data/` 内容仍是本地生成文件。

前面的自定义 prompt 评测命令默认只写 prediction 和 metric；如需在
`metric_logs/test.jsonl` 中记录 trace，额外加 `--log_trace`。

## Baselines

每个“算法 × benchmark”组合对应一个独立命令。当前 repo 的覆盖关系如下；
`—` 表示没有实现该组合：

| 算法 | HotpotQA | WebArena | VeruSAGE |
| --- | --- | --- | --- |
| Zero-shot | ✓ | 使用原始 prompt JSON | 使用原始 prompts |
| GEPA | ✓ | ✓ | ✓ |
| GEPA-SGD | ✓ | — | ✓ |
| Naive few-shot | ✓ | ✓ | — |
| Teacher few-shot | ✓ | — | — |
| Trace-RAG | ✓ | ✓ | ✓ |
| TextGrad | ✓ | ✓ | ✓ |

下面按 benchmark 给出每个已实现组合的命令。

### HotpotQA

在 repo 根目录先设置公共变量：

```bash
BASE_RUN=experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_Baseline_gpt-4o
ARTIFACT_DIR=experiment_runs_data/baseline_artifacts
LM_CONFIG='{"name":"gpt-4o","model":"openai/gpt-4o","api_key":"env:OPENAI_API_KEY","temperature":0.0}'
```

原始 zero-shot prompt 在 `$BASE_RUN/metric_logs/prompts/`。其余 baseline
从已提交的 training trace 生成新 prompt：

#### Zero-shot Baseline

```bash
.venv/bin/python -m scripts.run_experiments \
  --bm_idx 0 --benchmark_name HotpotQABench \
  --program_idx 0 --prog_name HotpotMultiHop \
  --opt_idx 0 --optim_name Baseline \
  --lm_config "$LM_CONFIG" --num_threads 1 --seed 0
```

#### GEPA baseline

要修改的 prompt 由 `scripts/experiment_configs.py` 中的
`HOTPOTQA_PREDICTOR_SUBSET` 指定。

```bash
.venv/bin/python -m scripts.run_experiments \
  --bm_idx 0 --benchmark_name HotpotQABench \
  --program_idx 0 --prog_name HotpotMultiHop \
  --opt_idx 1 --optim_name GEPA \
  --lm_config "$LM_CONFIG" --num_threads 1 --seed 0
```

#### Naive few-shot

直接选 training trace 中的模型输入和输出，不调用 teacher LM。

```bash
.venv/bin/python scripts/naive_fewshot_baseline.py \
  --baseline_run_dir "$BASE_RUN" \
  --output_prompts_dir "$ARTIFACT_DIR/hotpot_naive_fewshot" \
  --num_shots 3 --shuffle --seed 0 \
  --run_dir experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_NaiveFewShot_gpt-4o \
  --lm_config_json "$LM_CONFIG" --num_threads 1
```

#### Trace-RAG

先建一次 BM25 index，再为四个模块分别生成 prompt：

```bash
.venv/bin/python baseline_rag/trace_rag.py build-index \
  --traces_dir "$BASE_RUN/metric_logs/traces" \
  --prompt_dir "$BASE_RUN/metric_logs/prompts" \
  --output_dir "$ARTIFACT_DIR/hotpot_rag_index"

for ACTION in summarize1 create_query_hop2 summarize2 final_answer; do
  .venv/bin/python baseline_rag/trace_rag.py build-prompt \
    --index_dir "$ARTIFACT_DIR/hotpot_rag_index" \
    --prompt_dir "$BASE_RUN/metric_logs/prompts" \
    --action "$ACTION" \
    --output_path "$ARTIFACT_DIR/hotpot_rag/${ACTION}.md"
done

.venv/bin/python scripts/eval_custom_prompts.py \
  --prompts_dir "$ARTIFACT_DIR/hotpot_rag" \
  --run_dir experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_TraceRAG_gpt-4o \
  --lm_config_json "$LM_CONFIG" --num_threads 1
```

默认输出是“原 prompt + 检索证据”。如需 LM 将证据改写进 prompt，在
`build-prompt` 后增加
`--lm_model openai/gpt-4o --lm_api_key_env OPENAI_API_KEY`。

#### TextGrad-style

对 trace 生成 textual gradients，再一次性重写四个模块的 prompt。

```bash
.venv/bin/python scripts/textgrad_hotpotqa_baseline.py \
  --baseline_run_dir "$BASE_RUN" \
  --output_prompts_dir "$ARTIFACT_DIR/hotpot_textgrad" \
  --lm_config_json "$LM_CONFIG" \
  --only_incorrect \
  --max_traces 20 \
  --max_traces_per_prompt_in_apply 20 \
  --seed 0 \
  --run_dir experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_TextGrad_gpt-4o \
  --num_threads 1
```

去掉两个 `--max_*` 参数会使用所有符合条件的 trace，成本更高。

#### 评测任意一组 prompt

```bash
PROMPTS_DIR="$ARTIFACT_DIR/hotpot_textgrad"  # 换成任意生成目录
RUN_NAME=HotpotQABench_HotpotMultiHop_TextGrad_gpt-4o

.venv/bin/python scripts/eval_custom_prompts.py \
  --prompts_dir "$PROMPTS_DIR" \
  --run_dir "experiment_runs_data/experiment_runs/seed_0/${RUN_NAME}" \
  --lm_config_json "$LM_CONFIG" \
  --num_threads 1
```

已有示例产物在 `scripts/prompts_naive_fewshot/`、
`scripts/prompts_teacher_fewshot/` 和 `scripts/prompts_revised/`。新生成的
内容默认放在被 Git 忽略的 `experiment_runs_data/baseline_artifacts/`；
确认后再复制到 `scripts/prompts_<name>/` 提交。

### VeruSAGE

`verusage/` 和 trace 是本地输入，先按 `verusage/README.md` 安装依赖并
准备好：

```bash
TRACE_DIR=/path/to/verusage/traces
VERUSAGE_PROMPTS=verusage/agents/prompts
```

#### GEPA baseline

```bash
.venv/bin/python scripts/verusage_offline_prompt_opt.py \
  --traces_dir "$TRACE_DIR" \
  --prompt_dir "$VERUSAGE_PROMPTS" \
  --output_dir "$ARTIFACT_DIR/verusage_gepa" \
  --prompt_subset assertion_reasoning_pipeline,instantiate_forall \
  --lm_model openai/gpt-4o \
  --lm_api_key_env OPENAI_API_KEY
```

#### Trace-RAG

```bash
.venv/bin/python baseline_rag/trace_rag.py build-index \
  --traces_dir "$TRACE_DIR" \
  --prompt_dir "$VERUSAGE_PROMPTS" \
  --output_dir "$ARTIFACT_DIR/verusage_rag_index"

.venv/bin/python baseline_rag/trace_rag.py build-prompt \
  --index_dir "$ARTIFACT_DIR/verusage_rag_index" \
  --prompt_dir "$VERUSAGE_PROMPTS" \
  --action instantiate_forall \
  --output_path "$ARTIFACT_DIR/verusage_rag/instantiate_forall.md"
```

#### Full-trace TextGrad

```bash
.venv/bin/python scripts/textgrad_fulltrace_baseline.py \
  --traces_dir "$TRACE_DIR" \
  --prompt_dir "$VERUSAGE_PROMPTS" \
  --base_prompts postcondition_repair,induction \
  --output_dir "$ARTIFACT_DIR/verusage_textgrad" \
  --lm_model openai/gpt-4o \
  --lm_api_key_env OPENAI_API_KEY \
  --max_traces 10
```

`--base_prompts` 决定生成哪些 revised prompts。运行 VeruSAGE 时，通过
`verusage/main.py --prompts-dir <生成目录>` 加载它们。

### WebArena

输入 trace 应放在 `webarena/<site>/traces/`。下面以 `shopping` 为例。

#### GEPA baseline

```bash
.venv/bin/python webarena/run_gepa_webarena_offline.py \
  --prompt_json webarena/p_cot_id_actree_2s.json \
  --traces_dir webarena/shopping/traces \
  --output_json webarena/shopping/p_cot_id_actree_2s_gepa.json \
  --lm_model openai/gpt-4o \
  --lm_api_key_env OPENAI_API_KEY \
  --filter_by_score failed --max_samples 10
```

#### Naive few-shot

```bash
.venv/bin/python webarena/build_webarena_baselines.py \
  --sites shopping --baselines naive_fewshot \
  --max_fewshot_examples 3
```

#### RAG

```bash
.venv/bin/python webarena/build_webarena_baselines.py \
  --sites shopping --baselines rag --rag_top_k 8
```

#### TextGrad

```bash
.venv/bin/python webarena/build_webarena_baselines.py \
  --sites shopping --baselines textgrad \
  --lm_config_json "$LM_CONFIG" \
  --textgrad_max_traces 10
```

输出为
`webarena/<site>/p_cot_id_actree_2s_<baseline>.json`。可先加 `--dry_run`
检查输入；本 repo 只负责离线生成 prompt JSON，浏览器环境中的正式评测需
使用 WebArena runner。

### GEPA 适配说明

GEPA 做了任务适配：

- 显式传入本次允许修改的 prompt 集合，不默认修改全部 prompt。
- 使用各任务已有的离线 trace 和反馈。
- 只替换指定 prompt，其他 prompt 和程序结构保持不变。

repo 中有三个适配入口：

| 任务 | 入口 | 如何指定 prompt |
| --- | --- | --- |
| HotpotQA | `scripts/run_experiments.py` | `scripts/experiment_configs.py` 中的 `HOTPOTQA_PREDICTOR_SUBSET` |
| WebArena | `webarena/run_gepa_webarena_offline.py` | 固定只优化 prompt JSON 的 `intro` |
| VeruSAGE | `scripts/verusage_offline_prompt_opt.py` | `--prompt_subset prompt_a,prompt_b` |

当前显式配置的四个 predictor 是：

```python
HOTPOTQA_PREDICTOR_SUBSET = [
    "summarize1.predict",
    "create_query_hop2.predict",
    "summarize2.predict",
    "final_answer.predict",
]
```

结果位于：

```text
experiment_runs_data/experiment_runs/seed_<seed>/
  HotpotQABench_HotpotMultiHop_<optimizer>_<model>/
```

优化后的 DSPy program（包含 revised prompts）保存在
`evaluation_results/optimized_program/`；运行日志和候选 prompt 信息保存在
同一个 run 目录。

WebArena prompt JSON 包含 `intro`、`examples`、`template` 和
`meta_data`。这个适配只把 `intro` 交给 GEPA，生成后再写回原 JSON。
VeruSAGE 的 `--prompt_subset` 和 GEPA-SGD 的 `--prompt_names` 都是显式传入
要修改的 prompt；未列出的 prompt 不会改变。

## Local-only inputs

Large or generated inputs are intentionally excluded from Git:

- `hotpotQA/`
- `wiki.abstracts.2017.tar.gz`
- `gepa_artifact/benchmarks/hover/wiki.abstracts.2017.jsonl`
- `gepa_artifact/benchmarks/hover/bm25s_retriever/`
- `baseline_rag/*_index*/`
- `baseline_rag/*_output*/`
- `webarena/**/traces/`
- 除上述 HotpotQA Baseline trace 外的 `experiment_runs_data/`
- evaluation outputs, other metric logs, and DSPy caches

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
retained, except for the HotpotQA Baseline traces documented above.
See [`BASELINES.md`](BASELINES.md) for a short baseline inventory.
