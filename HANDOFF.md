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
/home/v-yingchang/gepa-artifact/.venv/bin/python scripts/eval_custom_prompts.py \
  --prompts_dir /home/v-yingchang/gepa-artifact/scripts/prompts_revised \
  --run_dir /home/v-yingchang/gepa-artifact/experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_CustomPrompts_gpt-4o3_0122 \
  --num_threads 1
```

脚本默认复用 Baseline 的 `config.json` 中的 LM 配置；评测结果、日志和
实际使用的 prompt 会写入 `run_dir`。

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
