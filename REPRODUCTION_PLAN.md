# GEPA reproduction plan

Reproduction of *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* ([arXiv 2507.19457](https://arxiv.org/abs/2507.19457)) using this artifact repository.

- **Where:** all runs happen on **Snellius**. Laptops are only for analysis and figures; B did Phase 2 on one. The Mac (M2) is also the fallback if the SBU budget runs low.
- **Budget:** shared course account `gpuuva090`, about 980k SBU left on 2026-11-04, shared with the other course users. **It expires 2026-12-18.** My own share is still to be confirmed (step 0.3).
- **External deadline:** `gpt-4.1-mini` is reported to leave the OpenAI API on **2027-01-14** (F16).
- **Team:** **A** (fork owner) and **B** (teammate) share the work. §6 has the split and the branch workflow; results go in §7.
- **Current scope (2026-11-04):** Table 1 on HotpotQA, HoVer, IFBench, AIME-2025 and LiveBench-Math, with Baseline, MIPROv2, GEPA and GEPA+Merge. No PUPA and no GRPO until there's an OpenAI key and a scope decision (step 5.3). Estimated ~14.5k SBU (±2×) plus ~0.5k for pilots, and €0 of API.
- **Legend:** ☐ todo · ◐ in progress · ☑ done · ✗ dropped. F1, F2, … refer to the findings in §3. Costs are in SBU (free course budget, but keep them low) unless marked `€`, which means out-of-pocket OpenAI API spend.

_Last updated: 2026-11-04, after B checked the plan against the authors' logs, caches and code (§3 group F, §8)._

---

## 1. Steps

Order: set everything up (Phases 0–3), run small pilots that measure the real cost (Phase 4), turn the measurements into an estimate and decide the scope (Phase 5), and only then start the main runs (Phases 6–7).

### Where each phase stands (after B's check, 2026-11-04)

| Phase | Status | What's done, what's left |
|---|---|---|
| 0. Housekeeping | ◐ | **Done:** branch pushed; git identity and local excludes set. **Left:** collaborator access, `.gitignore` of generated files (0.2), the upstream remote, and the SBU-share question. W&B will only be needed for GRPO once `GEPA_USE_WANDB=0` is merged (F13). |
| 1. Environment | ☐ | Nothing set up on Snellius yet. Pin the main environment to the exact versions of the original runs (F7) instead of a date cut-off. |
| 2. Reference data | ☑ | **Done by B on a laptop:** thinking mode was on (F25); run durations measured (F37); cost and token table for the GPT runs (§8); target numbers checked (F34). **No need to redo on Snellius,** except copying the shipped cache there for the replay check (4.0). |
| 3. Pilot tooling | ◐ | **Done:** 3.1a (`GEPA_EXPERIMENT_DIR`, on B's `feat/replay-and-analysis`). **Left:** 3.1b and 3.2–3.5. Draft job scripts exist in `scripts/snellius/`, but they start Arbor from the wrong environment and use the 4-GPU YAML; B's 3.3 replaces them. |
| 4. Pilots | ☐ | All still needed. Pilot 4.3 is the only source of Qwen tokens per call (F33). |
| 5. Estimate and decide | ☐ | After the pilots. |
| 6. Main runs | ☐ | The Baselines come from pilot 4.3, and MIPROv2 on IFBench from pilot 4.5. |
| 7. Extensions | ☐ | Extra seeds (7.4) become the first extension (F27). |
| 8. Analysis | ◐ | **Done:** 8.1, by B with the scripts in `analysis/` on the reference data (§8). **Left:** 8.2 and 8.3, which wait for our runs. |

### Today (2026-11-04): first Table 1 runs, without PUPA and GRPO

Goal for today:
- environments ready for both of us;
- pilot tooling merged into `reproduction`;
- pilots 4.1–4.3 finished, which fills the Baseline row of Table 1 for 5 benchmarks;
- pilots 4.4 and 4.5 running overnight.

Budget for today is about 1,000–1,100 SBU and €0. B has already done steps 2.1–2.4, 3.1a, the log analysis and the W&B switch on a laptop.

| # | Who | Task | Steps | Done when | Status |
|---|---|---|---|---|---|
| T1 | A | Access | Branch `reproduction` is pushed (commit `e47fd9e`). Left: add B as a collaborator on the GitHub fork, and step 0.2 | B can push | ◐ branch pushed |
| T2 | B | Own checkout on Snellius | `git clone` the fork, then `git switch reproduction` | B is on the branch | ☐ |
| T3 | A | Budget | Ask the account contact what each person's SBU share is | Answered | ☐ |
| T4 | A, then B | Pinned environment | A: 1.1–1.3, with `uv.lock` pinned to the versions of the original runs (F7). Then both run `uv sync --locked`. B does 1.3 only if A hasn't committed `uv.lock` yet | Same versions on both sides | ☐ |
| T5 | Both | Arbor environment and weights | 1.4 and 1.6 (repo id `qwen/qwen3-8b`), each in their own checkout. A needs them too, for A's own pilots | `import vllm` works; weights cached | ☐ |
| T6 | A | BM25 index | 1.7: a short CPU job. Only HotpotQA and HoVer use it | `bm25s_retriever/` exists | ☐ |
| T7 | A | Metric-call cap | 3.1b on `feat/pilot-caps`, with a PR that B reviews. Skip 2.1–2.4 and 3.1a, which are done | A capped run stops after N calls | ☐ |
| T8 | B | Job tooling | On `feat/slurm-tooling`, with a PR that A reviews: 3.2 one-GPU YAML; 3.3 SLURM template (Arbor from the Arbor environment, scratch directory, free port, wait until the server answers); 3.4 launch filter with the port filled in; 3.5 measurement hooks | Dry-run job works | ☐ |
| T9 | B | Share the laptop work | Push `feat/replay-and-analysis` (3.1a, `GEPA_USE_WANDB`, replay script, `analysis/`) and open the PR into `reproduction`, rebased on the latest plan | PR open | ☐ |
| T10 | B | Pilots 4.1 and 4.2 | Network check from a compute node; Baseline dry run into the pilot directory; record the server start-up time | Both pass | ☐ |
| T11 | A and B | Pilot 4.3: the Table 1 Baselines | Real directory. A: HotpotQA, HoVer. B: IFBench, AIME-2025, LiveBench-Math. Record tokens per call (from Arbor's response log, F33) and seconds per rollout in §7.3 | 5 scores in §7.1 | ☐ |
| T12 | A | Pilot 4.4 (overnight) | GEPA on HotpotQA with 1, 3 and 6 runs sharing one server. Needs T7 and T8 | Seconds per iteration in §7.3 | ☐ |
| T13 | B | Pilot 4.5 (overnight) | MIPROv2 on IFBench, as a real run | Wall-clock in §7.2 and §7.3 | ☐ |
| T14 | Both | Report | Fill in §7.1 and §7.3, then commit as `docs(results): ...` after `git pull --rebase`. After every pilot, grep `run_log.txt` for errors (F29) and check `budget-overview` | §7 up to date | ☐ |
| T15 | Both | Decide | A ~$1 GPT-4.1-mini pilot (4.7) if an OpenAI key exists; extra seeds (7.4) as the first extension (F27) | Decisions in §5 | ☐ |

### Phase 0: Housekeeping

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 0.1 | Git identity and local-only excludes | `git config --global user.name/user.email`. Add personal, local-only tool files to `.git/info/exclude` (never to `.gitignore`). `git remote add upstream https://github.com/gepa-ai/gepa-artifact` | `git status` lists only files meant for the repo | 0 | ◐ identity and excludes done; upstream remote to do |
| 0.2 | Ignore generated files | Add `wiki.abstracts.2017.tar.gz`, `gepa_artifact/benchmarks/hover/wiki.abstracts.2017.jsonl`, `gepa_artifact/benchmarks/hover/bm25s_retriever/`, `launch_commands`, `reference_runs_data/`, `storage/` to `.gitignore`. Commit as `chore: ignore generated retrieval and run files` (F9, F23) | Commit exists | 0 | ☐ |
| 0.3 | Accounts and budget decisions | Ask the account contact (see `accinfo`) what my SBU share is. Decide whether to get an OpenAI key (F14, F16). W&B: once `GEPA_USE_WANDB=0` is merged, only GRPO needs a key (F13). Keep keys in `~/.config/gepa/secrets.env` (chmod 600), sourced by job scripts, never in the repo | Share known. OpenAI decision made | 0 | ☐ |

### Phase 1: Environment (login node)

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 1.1 | Install uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` installs to `~/.local/bin`, which is already on PATH. uv also provides Python ≥3.11 (system Python is 3.9, F8) | `uv --version` | 0 | ☐ |
| 1.2 | Clone the forks | `bash setup_gepa_repo.sh` | `gepa_artifact/utils/{dspy,arbor}` exist on branch `gepa_study` | 0 | ☐ |
| 1.3 | Main environment, pinned to the original runs | Use the versions recorded by the original runs (F7): Python 3.12.9 (`uv python pin 3.12.9`), dspy 2.6.23 (the fork), litellm 1.74.0.post1, openai 1.93.2, pydantic 2.11.7, ujson 5.10.0, cloudpickle 3.1.1, datasets 3.6.0. For example, `uv lock --upgrade-package litellm==1.74.0.post1` for each package changes only `uv.lock`. Commit `uv.lock` and `.python-version` as `build: pin dependencies to the versions of the original runs` | `uv pip list` shows the pinned versions; `uv run python -c "from dspy.teleprompt.grpo import GRPO"` works | 0 | ☐ |
| 1.4 | Arbor/vLLM environment | `cd gepa_artifact/utils/arbor && uv sync --exclude-newer 2025-08-04`. A separate environment, because the default one has no `arbor-ai` (F6). The check found no recorded vLLM/torch versions, so the date cut-off stays here | `uv run python -c "import vllm"` in that directory | 0 | ☐ |
| 1.5 | Datasets and launch commands | `uv run python -m scripts.generate_launch_commands > launch_commands`. This builds every benchmark, which downloads all 6 datasets from the HF Hub (F17, F22) | 72 lines, no download errors | 0 | ☐ |
| 1.6 | Model weights | From the Arbor environment: `uv run huggingface-cli download qwen/qwen3-8b` (about 16 GB). Use the same repo-id spelling Arbor requests (F11) | Weights in the HF cache | 0 | ☐ |
| 1.7 | Build the BM25 index once | Short CPU job on `rome` (16 cores, about 28 GB RAM), run from the repo root, that calls `init_retriever()` from `gepa_artifact/benchmarks/hover/hover_program.py` (F23). Only HotpotQA and HoVer use it, so only A needs it | `gepa_artifact/benchmarks/hover/bm25s_retriever/` exists | ~10–30 | ☐ |

### Phase 2: Reference data (no GPU; done by B on a laptop)

This shows how the authors ran Qwen3-8B (thinking on, F25) and gives reference times and costs. It doesn't need to be repeated on Snellius.

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 2.1 | Get the authors' archive | On Snellius there's no git-lfs module: put the `git-lfs` binary from the linux-amd64 release (github.com/git-lfs/git-lfs/releases) in `~/.local/bin`, then `git lfs install && git lfs pull` | `sha256sum experiment_runs_data.tar.gz` = `0e1db10331bbbdd48701443b0690d5b7b8abef841a73b76882d26853b3f38dd7` | 0 | ☑ B (laptop) |
| 2.2 | Extract outside `BASE_EXPERIMENT_DIR` | `mkdir reference_runs_data && tar -xzf experiment_runs_data.tar.gz -C reference_runs_data`. **Never extract into `experiment_runs_data/`** (F19) | Authors' runs listed under `reference_runs_data/` | 0 | ☑ B (laptop) |
| 2.3 | Reference analysis | **Thinking:** read from the cached raw Qwen responses (`reasoning_content`). The metric logs store `prediction: null`, so they can't show it (F25). **Tokens:** only the GPT-4.1-mini runs report them; Arbor runs report 0 (F33). **Durations:** from the authors' logs (F37) | Thinking on. Token and cost table for the GPT runs. Run durations | 0 | ☑ B (laptop) |
| 2.4 | Target numbers | Check the reference `evaluation_result.txt` scores against the paper tables in §4 | 46 of 58 cells match, 1 differs, 11 have no run (F34) | 0 | ☑ B (laptop) |

### Phase 3: Pilot tooling (our code)

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 3.1 | Separate output directory and budget caps | (a) `GEPA_EXPERIMENT_DIR` overrides `BASE_EXPERIMENT_DIR` (default unchanged). (b) A flag that caps `max_metric_calls` (default unchanged). (c) Only if GRPO is in scope: the same kind of cap for GRPO's `num_train_steps`. Every dry run and pilot goes to a separate directory (F18, F21). Commit as `feat(scripts): add metric-call cap for pilots` | A capped GEPA run stops after N calls and writes nothing to the real directory | 0 | ◐ (a) done by B on `feat/replay-and-analysis`; (b) A on `feat/pilot-caps` |
| 3.2 | 1-GPU Arbor config | YAML with `inference.gpu_ids: '0'` and `training.gpu_ids: ''`, kept in the repo (e.g. `slurm/arbor_inference_1gpu.yaml`) (F2) | Committed | 0 | ☐ |
| 3.3 | SLURM job template | Start Arbor **from the Arbor environment (1.4)**, from a per-job scratch directory, on a free port (F9). Wait until it answers. Run one or more `run_experiments` processes against that `api_base`, then stop Arbor. Chain jobs with `--dependency=afterok` (F20). This replaces the draft scripts in `scripts/snellius/`, which use the wrong environment and the 4-GPU YAML. Commit as `feat(slurm): add Qwen3-8B job script` | Dry-run job succeeds | — | ☐ |
| 3.4 | Launch-command selection | Helper that filters `launch_commands` by benchmark, optimizer and LM, and fills in the `{portnum}` placeholder (F22) | Prints the chosen commands, ready to run | 0 | ☐ |
| 3.5 | Measurement hooks | In the job template, log `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 30` and the start/end time. After each job, record `sacct -j <id> --format=JobID,Elapsed,AllocTRES` and the SBU used (`budget-overview`) in a pilot log (e.g. `pilots/pilot_log.csv`). Qwen tokens: Arbor returns no usage (F33), so count them from Arbor's `storage/logs/<timestamp>/inference_responses.jsonl` (F9) with the Qwen3 tokenizer | Each pilot leaves one row: what ran, wall-clock, SBU, GPU utilization, tokens or iterations | 0 | ☐ |

### Phase 4: Pilots for the compute estimate (1× H100 unless noted)

Each pilot answers one question that the estimate in §2 depends on; the assumption it replaces is listed in §2.1.
- **Pure pilot overhead:** ~350–450 SBU, plus ~600 if GRPO is in scope.
- **Reused:** pilots 4.3 and 4.5 are real runs that count toward Phase 6.
- **After every pilot,** grep `run_log.txt` for errors: failed calls are scored as zero without stopping the run (F29).

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 4.0 | Does the evaluation code run on Snellius, before any GPU job? *(optional)* | Replay a reference Baseline from the shipped cache with `scripts/replay_test_eval.py` (B's `feat/replay-and-analysis`) in a CPU job, with no GPU and no key (F38). Needs that run's cache directory on Snellius | Replayed score equals the reference score | ~16 | ☐ |
| 4.1 | Can compute nodes reach W&B and HF? | In a short job: `curl -sI https://api.wandb.ai` and `curl -sI https://huggingface.co` (F13) | Both reachable | ~5 | ☐ |
| 4.2 | Does the pipeline work, and how long does the server take to start? | Baseline on HotpotQA with Qwen3-8B, `--dry_run`, **written to the pilot directory** (F18). Time from job start until Arbor answers | `evaluation_results/` exists in the pilot directory. Startup time recorded | ~30 | ☐ |
| 4.3 | How many tokens per call, and how fast at 32 threads? | All 5 OpenAI-free Baselines (HotpotQA, HoVer, IFBench, LiveBench-Math, AIME) in the **real directory**, split by owner. Keep Qwen3's default chat template and `max_tokens=16384` (F25). Count tokens from Arbor's response log, not from `evaluation_result.txt`, which will show 0 (F33) | Output tokens per call and seconds per rollout for each benchmark. Scores close to §4 | ~250–300 (real runs) | ☐ |
| 4.4 | How long is one GEPA iteration, and how many runs can share a GPU? | GEPA on HotpotQA, pilot directory, **without** `--use_cache_from_opt`, capped at ~600 metric calls. Run 1, then 3, then 6 processes against one server, about 30 min each. Ignore the first full val eval when timing. Iteration times come from the growth of `instruction_proposer_inpouts.jsonl` (one line per reflection, F35) or from W&B `_runtime` if W&B is on | Seconds per iteration and GPU utilization for 1, 3 and 6 runs, giving the packing factor | ~300–400 | ☐ |
| 4.5 | Does the MIPROv2 estimate hold? | The first real MIPROv2 run, on IFBench (shortest without OpenAI), after its Baseline from 4.3. Compare its wall-clock with §2 and with the authors' 1.1–4.7 h on 4 GPUs (F37) | MIPROv2 correction factor known | ~400 (real run) | ☐ |
| 4.6 | How long is one GRPO step? *(only if GRPO is in scope)* | GRPO environment, 3 H100, HotpotQA, pilot directory, 10–20 steps (cap from 3.1c) | Seconds per step and per val eval | ~600 | ☐ |
| 4.7 | What does a rollout cost in API money, and do rate limits bite? *(only with an OpenAI key)* | GPT-4.1-mini Baseline on IFBench (CPU job) and the Qwen3-8B PUPA Baseline, both in the real directory. Read the cost in the OpenAI usage dashboard. Grep `run_log.txt` for rate-limit errors (F29) | $ per rollout for both. No rate-limit errors, or a thread count that avoids them | ~20 + ~€1–2 (real runs) | ☐ |

### Phase 5: Estimate and decide

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 5.1 | Measured estimate | Put the pilot numbers into the cost model and update §2. Add GEPA's budget overshoot (F36). Mark which numbers are measured. Commit as `docs(plan): update cost estimate with pilot measurements` | §2 based on measurements | 0 | ☐ |
| 5.2 | Packing layout | Choose which runs share a GPU: up to the number found in 4.4, grouping runs of similar length. Write the job list with its dependencies | Job list with SBU per job | 0 | ☐ |
| 5.3 | Go/no-go and scope | Compare the total with our SBU shares (0.3) and the € budget. Decide extra seeds (7.4), GRPO (a subset?), PUPA and Table 2. Record the decisions in §5 | Decisions recorded | 0 | ☐ |

### Phase 6: Main Qwen3-8B runs without an OpenAI key (HotpotQA, HoVer, IFBench, LiveBench-Math, AIME)

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 6.1 | Baseline ×5 | Already done in pilot 4.3 | 5× `evaluation_results/` | (in 4.3) | ☐ |
| 6.2 | MIPROv2-Heavy, remaining 4 | IFBench was done in pilot 4.5. One 1-H100 job per benchmark | 5 done | ~2.4k (estimate before 5.1; could be 2–3× higher, F37) | ☐ |
| 6.3 | GEPA + GEPA-MERGE ×5 | Packed as decided in 5.2 (F3), `--dependency=afterok:<MIPRO job>` (F20). Adding Abl-SelectBestCandidate (Table 3) as a third process on HotpotQA, HoVer and IFBench costs ~650 SBU more; decide in 5.3 | 10 done | ~11.4k (estimate before 5.1, plus up to ~20% overshoot, F36) | ☐ |

### Phase 7: Extensions (decided in 5.3; 7.4 comes first)

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 7.4 | **First extension:** extra seeds for the key cells | `SEEDS` in `scripts/generate_launch_commands.py`. A seed other than 0 reshuffles train/val only (F24). Start with the cells the write-up's main claims rest on: one seed per cell is at noise level, and the artifact shows a 4.7-point swing between two runs of the same method (F27) | Chosen cells have ≥2 seeds | ×1 per seed | ☐ |
| 7.1 | PUPA with Qwen3-8B | Needs a real OpenAI key (F14). Same chain as Phase 6 | 5 runs done | ~1k + ~€27 (incl. SelectBest) | ☐ |
| 7.2 | GPT-4.1-mini runs (Table 2) | Real OpenAI key, after pilot 4.7. API-only, so 16-core CPU jobs on `rome` (F31). Must finish before 2027-01-14 (F16) | Up to 30 runs done | ~400 + ~€212–260 (~€255–315 with VAT; §2.3) | ☐ |
| 7.3 | GRPO | After pilot 4.6. GRPO environment (`pyproject_grpo.toml` in a separate checkout; `flash-attn` build needs a CUDA module). 3 H100s, `arbor_train.yaml` (F28). Cheapest useful subset: HotpotQA + IFBench (Figure 1) | The chosen benchmarks done | ~7k–18.5k per benchmark (~85k for all 6) | ☐ |

### Phase 8: Analysis and write-up

| # | Task | How | Done when | Cost | Status |
|---|---|---|---|---|---|
| 8.1 | Analysis of the reference runs | B's scripts in `analysis/` (on `feat/replay-and-analysis`) on `reference_runs_data/`: tables, budget curves, search trees, prompt sizes, generalization gap (§8). `scripts/generate_figures.ipynb` reproduces the paper's figures | Reference tables and figures | 0 | ☑ B (laptop) |
| 8.2 | Figures and tables from our runs | The same `analysis/` scripts and `scripts/generate_figures.ipynb` on our `experiment_runs_data/` (A's and B's runs combined, §6) | Figures generated | 0 | ☐ |
| 8.3 | Compare and document deviations | Compare against §4 and the reference runs. Record deviations: hardware, package versions, seeds, thinking mode (F25–F27). Report both the metric-call budget and the calls actually used (F36). Count reflection calls from our own `instruction_proposer_inpouts.jsonl` (F35). Check whether our search trees show the same pattern as the authors' (F39) | Write-up done | 0 | ☐ |

---

## 2. Cost per table and variant

- **SBUs** come from the course budget, so they're free for me, but the budget is shared and I want to use as few as possible.
- **OpenAI API costs** are the only out-of-pocket money. They're shown in € at **€0.88 per $**. OpenAI bills in $ and may add **21% Dutch VAT** on a private account.
- **Uncertainty:** SBU numbers are a first-principles estimate, ±2×. API numbers are ±25%. Both get replaced with measured numbers in step 5.1, after the pilots in Phase 4.
- **Order of runs:** within a table, a GEPA-family run needs the MIPROv2 run, and MIPROv2 needs the Baseline (F20). So reproducing a GEPA row means paying for those two rows as well.

### 2.1 What the numbers are based on

**Known exactly** (from the code and the paper)
- Rollout budgets per benchmark: Table 1 / `known_max_calls`.
- GRPO schedule: 500 steps × 48 rollouts, plus a val eval every 20 steps.
- LM calls per rollout: HotpotQA 4, HoVer 4, IFBench 2, PUPA 2 local plus OpenAI calls, math 1.
- Concurrency: 32 threads for evaluations, minibatches of 3 in GEPA.
- The paper's API costs for the GPT-4.1-mini runs (appendix E.3).
- Prices: H100 = 192 SBU/h. The smallest CPU job is 16 cores = 16 SBU/h (F31). `gpt-4.1-mini` costs $0.40 per 1M input tokens and $1.60 per 1M output tokens.

**Assumed** (the step that measures each one is in brackets)
- Qwen3 output tokens per call with thinking on (F25): multi-hop modules 700, IFBench 1,200, PUPA 600, math 5,000. [4.3 only: the reference runs have no Qwen token counts, F33]
- Qwen3-8B decode speed on 1× H100: ≈100 tok/s for one stream, ≈2,250 tok/s across 32 streams. [4.3, 4.4]
- A GEPA iteration is 2 minibatches plus 1 reflection call, and GEPA spends ≈40% of its budget on full val evals. [4.4]
- Packing 2–3 GEPA runs on one server slows each one by ≈15%. [4.4]
- MIPROv2 reaches ≈70% of full throughput. [4.5]
- GRPO LoRA training takes ≈0.34 s per 1k tokens per sequence. [4.6]
- Each PUPA rollout makes 4 OpenAI calls (untrusted model, 2 quality judgments, 1 leakage count; F14). Together they come to ≈3.4k input + 0.8k output tokens, about $0.0027. PUPA's splits are 111 train / 111 val / 221 test (F24). [4.7]

### 2.2 Table 1: Qwen3-8B (6 benchmarks, 1 H100 per job)

| Variant | Needs first | SBU | API € | Why |
|---|---|---|---|---|
| Baseline | – | ~300 | ~0.5 | Up to 300 test rollouts per benchmark (AIME 150, PUPA 221) on 32 threads, so short; mostly GPU startup. API: PUPA only, 221 rollouts × $0.0027 |
| MIPROv2 | Baseline | ~3,100 | ~6.5 | 1,839–7,051 rollouts per benchmark in batches of 35, which keeps the GPU busy. API: PUPA's 2,426 + 221 rollouts |
| GEPA | MIPROv2 | ~10,600 | ~6.5 | Same rollout budget as MIPROv2, but it works on 3 examples at a time and makes one reflection call at a time, so the GPU mostly waits (F3). HotpotQA alone is ≈690 iterations × ≈70 s ≈ 15 h |
| GEPA+Merge | MIPROv2 | **~1,600** in the same job as GEPA (~10,600 alone) | ~6.5 | Shares GEPA's server; the second process only adds ≈15% wall-clock |
| GRPO | – | ~84,600 | ~65 | 3 GPUs × 12–32 h per benchmark: 24,000 training rollouts, a LoRA update every 48 rollouts, and 25 val evals. PUPA's reward calls the OpenAI judge for each of ≈27,000 rollouts (24,000 train + 25 × 111 val + 221 test) |
| **Total** | | **~15,600** without GRPO, **~100,200** with | **~20** without GRPO, **~85** with | |

SBU per benchmark:

| Variant | HotpotQA | HoVer | IFBench | PUPA | AIME-2025 | LiveBench-Math |
|---|---|---|---|---|---|---|
| Baseline | 49 | 49 | 46 | 37 | 47 | 64 |
| MIPROv2 | 776 | 793 | 417 | 223 | 436 | 436 |
| GEPA | 2,833 | 2,906 | 1,335 | 673 | 1,428 | 1,446 |
| GEPA+Merge (same job as GEPA) | 425 | 436 | 200 | 101 | 214 | 217 |
| GRPO | 18,461 | 18,461 | 10,573 | 7,018 | 14,382 | 15,754 |

Cross-check with the authors' run times (F37), and one known addition:
- **GEPA:** it took 1.2–6.4 h per run on 4-GPU servers. One GPU makes each call about 1.8× slower, which suggests at most ~12 h. So the ~15 h for HotpotQA above is on the safe side.
- **GRPO:** 12–24 h on 3 GPUs, which matches.
- **MIPROv2:** 1.1–4.7 h on 4 GPUs. It keeps the GPU busy, so on one GPU it may take up to 2–3× longer than assumed above. Pilot 4.5 settles this.
- **Not included above:** GEPA-family runs use 2–23% more metric calls than their budget (F36).

### 2.3 Table 2: GPT-4.1-mini (6 benchmarks, API only, 16-core CPU jobs)

| Variant | Needs first | SBU | API € | Why |
|---|---|---|---|---|
| Baseline | – | ~12 | ~5 | ≈1,400 test rollouts over the 6 test sets, at $0.002–0.004 each (bottom-up) |
| MIPROv2 | Baseline | ~55 | ~67 | The paper reports $76 for all 6 benchmarks (appendix E.3). Bootstrapped few-shot demos make prompts longer, so cost is mostly input tokens |
| GEPA | MIPROv2 | ~290 | ~76 | The paper reports $86. My bottom-up check gives $95, and HotpotQA + HoVer account for about $64 of it. Those programs make 4 calls per rollout with about 1.8k input tokens each, over a ~7k-rollout budget |
| GEPA+Merge | MIPROv2 | **~30** in the same CPU job as GEPA (~290 alone) | ~59 | The paper reports $67. Cheaper than GEPA, plausibly because merge steps combine existing prompts without a reflection call |
| GEPA-Qwen-Opt | Table 1 GEPA | ~12 | ~5 | Test-set eval of the Qwen3-8B-optimized programs with GPT-4.1-mini. Needs a small extra script (F30) |
| **Total** | | **~400** | **~212** from the paper's per-variant figures; **~260** (≈ $296, ~€315 with VAT) by the artifact's own accounting of the shipped GPT runs (§8) | Budget for the higher number. One GEPA run costs $6–25. Plus about €2–5 for pilots |

- **CPU time:** these runs mostly wait on the API, so the cost is small whatever the run time.
- **Rate limits (F29):** if pilot 4.7 shows rate-limit errors, lower `--num_threads`. That costs a few extra CPU SBU but avoids silently wrong scores.
- **Not reproducible here:** Trace, TextGrad and MIPROv2-No-Demos aren't in this repo (F30).

### 2.4 Table 3: candidate selection, Qwen3-8B (HotpotQA, IFBench, HoVer, PUPA)

| Variant | Needs first | SBU | API € | Why |
|---|---|---|---|---|
| Baseline | – | 0 | 0 | Reused from Table 1 |
| GEPA | – | 0 | 0 | Reused from Table 1 |
| SelectBestCandidate | Table 1 MIPROv2 | **~700** as a third process in the Table 1 GEPA jobs (~7,700 alone) | ~6.5 | `Abl-SelectBestCandidate`: same budget as GEPA, only the candidate selection differs. API: PUPA only |
| BeamSearch | – | – | – | Not in this repo (F30) |

### 2.5 Figures

- **Figure 1** (learning curves) comes from the Table 1 logs, but it needs GRPO on HotpotQA and IFBench: ~29,000 SBU, €0 API.
- **Figures 2–6** come from the runs above at no extra cost.
- **Figures 7–8** aren't in this repo.

### 2.6 Summary and how to keep SBUs low

| Scope | SBU | API € |
|---|---|---|
| Table 1 without GRPO, plus Table 3 (packed) | ~16,300 | ~27 |
| Table 2 | ~400 | ~212–260 (~255–315 with VAT) |
| GRPO for HotpotQA + IFBench (Figure 1) | ~29,000 | 0 |
| GRPO for the other 4 benchmarks | ~55,600 | ~65 (PUPA judge) |
| **Everything** | **~101,300** | **~304–352** (~368–426 with VAT) |

Ways to keep SBUs low, biggest saving first:
1. **Decide GRPO's scope.** It's 84% of all SBUs. The HotpotQA + IFBench subset covers Figure 1 for 29k.
2. **Run GEPA, GEPA+Merge and SelectBestCandidate in one job per benchmark.** That saves about 16k SBU compared with separate jobs.
3. **Use one GPU per server** instead of the README's four (F2). Otherwise everything costs about 4× more.
4. **Chain jobs with `--dependency=afterok`**, so no job sits on a GPU waiting for MIPROv2 (F20).
5. **Smoke-test with the metric-call cap in a separate directory** (step 3.1), never with a full run.

Add about 25% contingency for pilots and reruns.

### 2.7 For comparison: running the default matrix as is

"As is" means all 72 generated commands, each run in its own job, with the authors' resource layout:
- 4-GPU tensor-parallel Arbor servers for every Qwen run except GRPO.
- 3 GPUs for GRPO.
- GPT-4.1-mini runs on 16-core CPU jobs.

The SBU numbers assume a 4-GPU server is about 3× faster than one GPU for throughput-bound runs and about 1.8× faster for latency-bound runs.

| Variant | SBU, default layout | SBU, this plan | API €, Qwen3-8B (PUPA) | API €, GPT-4.1-mini |
|---|---|---|---|---|
| Baseline | ~1,100 | ~300 | ~0.5 | ~5 |
| MIPROv2-Heavy | ~5,300 | ~3,100 | ~6.5 | ~67 |
| GEPA | ~24,100 | ~10,600 | ~6.5 | ~76 |
| GEPA-MERGE | ~24,100 | ~1,600 (packed) | ~6.5 | ~59 |
| Abl-SelectBestCandidate | ~24,100 | ~1,000 (packed, 6 benchmarks) | ~6.5 | ~76 (not in any paper table) |
| GRPO | ~84,600 | ~84,600 | ~65 | 0: fails at start (F32) |
| GPT-4.1-mini CPU jobs, all variants | ~940 | ~410 | | |
| **Total** | **~164,000** (~79,500 without GRPO) | **~101,600** (~17,000 without GRPO) | **~92** | **~283** |

- **API spend for the full default matrix:** about **€375**, or about **€455 with VAT**. With 25% contingency, about €470 (≈ €565 with VAT).
- **Not counted above:** two traps that would burn SBUs for nothing in the default setup.
  - Submitting every command at once leaves the GEPA-family jobs idle on 4 GPUs while they wait for MIPROv2 (F20): about +17,000 SBU.
  - Following the `num_gpus = 4` metadata (F1) for the GPT-4.1-mini runs would put API-only runs on GPU nodes: about +44,000 SBU.
- **F32. The `GRPO × gpt-41-mini` commands can't work.** GRPO calls `lm.reinforce()` before its first validation (`dspy/teleprompt/grpo.py` in the fork), and the OpenAI provider isn't reinforceable. So these 6 commands fail with an assertion before any API call.
  → Drop them from the selection. The same goes for `GPT × Abl-SelectBestCandidate` unless I want it: it costs about €76 and appears in no paper table (Table 3 is Qwen3-8B only).

---

## 3. Findings and implications

Findings from reading the code, the forks (`gepa_study` branches) and the paper, and from inspecting the Snellius account on 2026-11-04. B checked them against the authors' logs, caches and code the same day:
- **Confirmed as written:** F1, F3, F5, F10, F12, F14, F15, F17, F18–F24, F27, F28, F29, F30, F32.
- **Corrected:** F7, F13 and F25, and F27/F29 got additions. These are marked *Check (B)*.
- **Not checked:** the rest, which are about Snellius, Arbor or OpenAI specifics.
- **New findings from the check:** F33–F39, in group F.

### A. Hardware and compute

- **F1. GPU requirements in the launch-command generator are unused.** `requires_80_gb_gpu` and `num_gpus` are computed in `scripts/generate_launch_commands.py:26-27` but never written into the commands. They're leftovers from the authors' scheduler.
  → Pick resources in the SLURM script. 80 GB only matters for GRPO training: H100 (94 GB) is fine, A100 (40 GB) is not.
- **F2. Arbor shards the model across every GPU in its YAML.** It launches vLLM with `--tensor-parallel-size <number of gpu_ids>`, `--gpu-memory-utilization 0.9` and prefix caching (Arbor `inference_manager.py`). `arbor_inference.yaml` lists 4 GPUs.
  → Qwen3-8B (about 16 GB of weights) fits on one H100, so 4 GPUs cost 4× the SBU for a modest latency gain. Use a 1-GPU YAML (step 3.2).
- **F3. GEPA is limited by latency, not throughput.** Each iteration runs a minibatch of only 3 examples (`num_dspy_examples_per_gepa_step=3`, `gepa_artifact/gepa/gepa.py:56`), then one sequential reflection call, then the minibatch again. Full 300-example val evals happen only when a candidate improves.
  → The GPU is mostly idle during GEPA. Run GEPA, GEPA-MERGE and the ablation together against one vLLM server; continuous batching serves them at almost single-run speed.
  → MIPROv2 evaluates minibatches of about 35 with 32 threads, so it uses the GPU much better.
- **F4. The inference server is launched lazily with a 32,768-token context.** When the first chat request arrives and no server is running, Arbor starts vLLM with `max_context_length: 32768` (Arbor `api/routes/inference.py:27`). The `LAUNCH_KWARGS` 8192 from `experiment_configs.py` is not used on this path. DSPy requests `max_tokens=16384` (`scripts/run_experiments.py:59`).
  → An H100 has plenty of KV-cache room. On an A100 (40 GB), long contexts would sharply limit concurrency.
- **F5. Billing and budget.** H100 costs 192 SBU/GPU-h, A100 128 SBU/GPU-h, CPU 1 SBU per core-hour (`scontrol show partition`). H100 is cheaper per generated token: about 1.5× the price for about 2× the decode bandwidth. The account is a shared course budget that expires 2026-12-18 (`accinfo`).
  → Estimate before every phase. Pilot first. Confirm my share.
- **F31. CPU allocation granularity.** On `rome`, an 8-core request is rounded up to 16 cores (`sbatch --test-only`), and memory is 1,792 MB per core. So the smallest CPU job costs 16 SBU/h and has about 28 GB RAM. GPU jobs on `gpu_h100` get 16 cores and 180 GB RAM per GPU by default (`DefCpuPerGPU=16`, `DefMemPerGPU=184320`).
  → Pack several API-bound runs into one 16-core job.

### B. Environment and dependencies

- **F6. The default environment can't serve Qwen.** The default `pyproject.toml` doesn't include `arbor-ai`, and so no vLLM. Only `pyproject_grpo.toml` adds it, along with `flash-attn`.
  → Every Qwen3-8B run needs an Arbor environment, not just GRPO. Use a separate environment in `gepa_artifact/utils/arbor` (step 1.4).
  → `mv pyproject_grpo.toml pyproject.toml` rewrites tracked files. Do it only in a separate checkout, and only for GRPO.
- **F7. No lockfile anywhere.** Neither the repo nor the Arbor fork has a `uv.lock` (the DSPy fork has its own). The forks were last changed on 2025-07-03 (Arbor) and 2025-07-11 (DSPy); the artifact's first commit is from 2025-08-04. Resolving today would pull vLLM, TRL and transformers versions newer than the forks.
  → *Check (B):* every original run recorded its exact packages in `<run>/wandb/wandb/run-*/files/requirements.txt`: Python 3.12.9, dspy 2.6.23, litellm 1.74.0.post1, openai 1.93.2, pydantic 2.11.7, ujson 5.10.0, cloudpickle 3.1.1, datasets 3.6.0.
  → Pin the main environment's `uv.lock` to these instead of using a date cut-off (step 1.3), and commit it. The Arbor environment has no recorded versions, so it keeps `--exclude-newer 2025-08-04` (step 1.4).
- **F8. Login-node facts.** System `python3` is 3.9 (the project needs ≥3.11, and uv provides it). There's no `uv` and no `git-lfs` module; `tmux` is at `/usr/bin/tmux`.
  - Home: 200 GB / 1M inodes.
  - Scratch (`/scratch-shared` → `/gpfs/scratch1/shared`): 8 TB / 3M inodes. I couldn't confirm its retention policy.
  → Keep results and environments in home. Use scratch only for things that can be re-downloaded, and for Arbor's logs.
- **F9. Arbor logs every request and response.** It appends them to `./storage/logs/<timestamp>/inference_{requests,responses}.jsonl`, relative to the directory it's started from (Arbor `cli.py:19`, `core/config.py:25`).
  → For thousands of rollouts this grows to gigabytes. Start Arbor from a per-job scratch directory, and gitignore `storage/`.
- **F10. `ArborRunner` is dead code.** It needs an `env.sh` that isn't in the repo, and `run_experiments.py:234` resolves its YAML as `utils/arbor/...` relative to the working directory, while the fork lives at `gepa_artifact/utils/arbor/`. It never runs anyway, because `launch_arbor` is forced to `False` (`run_experiments.py:160`).
  → Start Arbor yourself in the job script.
- **F11. Repo-id spelling for the model cache.** Arbor strips the `openai/arbor:` prefix and asks vLLM for `qwen/qwen3-8b`.
  → Pre-download with exactly that string so the HF cache hits. Otherwise the job may re-download about 16 GB.

### C. API keys and external services

- **F12. Both keys are required at startup.** `run_experiments.py:486-487` asserts that `OPENAI_API_KEY` and `WANDB_API_KEY` are set. The Qwen config sends its own dummy key to Arbor.
  → For Qwen3-8B on HotpotQA, HoVer, IFBench, LiveBench-Math and AIME, `OPENAI_API_KEY=dummy` is enough.
- **F13. W&B is mandatory for GEPA and GRPO.** GEPA always sets `use_wandb=True` (`run_experiments.py:301`) and calls `wandb.login(key=..., verify=True)` (`gepa_artifact/gepa/gepa_utils.py:374`). GRPO reports to W&B (`report_to: "wandb"`). Baseline and MIPROv2 don't use W&B.
  → Get a free account (step 0.3). Compute nodes need outbound internet; verify in step 4.1.
  → *Check (B):* B's `feat/replay-and-analysis` adds an opt-in `GEPA_USE_WANDB=0` that skips W&B for GEPA. After that merge, only GRPO needs a W&B key, so W&B accounts no longer block the pilots.
- **F14. PAPILLON (PUPA) can't run without OpenAI.** Both the program's "untrusted" model and the LLM judge are `openai/gpt-4.1-mini` (`gepa_artifact/benchmarks/papillon/__init__.py:9`, `papillon_utils.py:45`). These use the undated alias, while the task-LM config pins `gpt-4.1-mini-2025-04-14`.
  → No key means no PUPA; after the model's retirement it can't be reproduced as specified. The judge cost is small (tens of dollars at most).
  → **How a PUPA rollout works** (`papillon/papillon_program.py`, `papillon_utils.py`):
    1. The local model (`craft_redacted_request`, ChainOfThought) rewrites the private query into a request without PII.
    2. The untrusted `gpt-4.1-mini` answers that request. This is **OpenAI call 1**.
    3. The local model (`respond_to_query`) writes the final answer from the private query plus the external answer.
    4. The metric asks `gpt-4.1-mini` whether the answer is at least as good as the reference, twice with A/B swapped (**calls 2–3**). The answer counts as good unless the judge prefers the reference in both orders.
    5. It also counts how many of the example's PII units appear in the redacted request (**call 4**).
    6. Score = (quality + (1 − leakage fraction)) / 2.

    Only steps 1 and 3 are optimized. These OpenAI LMs use DSPy defaults (temperature 0, cache on, 3 retries, max 1,000 output tokens), so identical calls are served from the cache and cost nothing. That includes GEPA's feedback function, which re-runs the same judge.
- **F15. GEPA reflects with the task model.** `teacher_lm` is never set, so the reflection LM is the task LM (`gepa_artifact/gepa/gepa.py:542`). MIPROv2 likewise uses the task LM to propose prompts.
  → Qwen3-8B runs reflect on their own outputs, so no OpenAI calls happen outside PUPA.
- **F16. `gpt-4.1-mini` is being retired.** Secondary sources report OpenAI will remove GPT-4o, 4o-mini, 4.1 and 4.1-mini from the API on 2027-01-14 (not yet checked against OpenAI's deprecations page). The paper (appendix E.3) reports costs for all benchmarks: GEPA $86, GEPA+Merge $67, MIPROv2 $76. All of Table 2 came to under $500, including Trace and TextGrad, which aren't in this repo.
  → Table 2 and PUPA are only possible until mid-January 2027, so decide early.
  → OpenAI's own model page already marks `gpt-4.1-mini-2025-04-14` as **Deprecated** (checked 2026-11-04). Only the exact shutdown date comes from secondary sources.
- **F17. External data sources.**
  - HotpotQA creates a ColBERTv2 client for a remote server (`hotpot_program.py:7`) but actually retrieves with HoVer's local BM25 `search` (`hotpot_program.py:5`). There's no dependency on that server.
  - Datasets come from the HF Hub: `hover` and `hotpot_qa` (both `trust_remote_code=True`), `Columbia-NLP/PUPA`, `livebench/math`, `AI-MO/aimo-validation-aime`, and `MathArena/aime_2025`. The Wikipedia abstracts also come from HF.
  → Download everything on the login node first (step 1.5).
- **F29. API errors silently become wrong answers.** `create_lm` sets `num_retries: 0` and `max_tokens: 16384` (`scripts/run_experiments.py:59-60`). The evaluators tolerate many errors (`max_errors` = 100× the set size, `gepa_artifact/gepa/gepa.py:308,325`; 10× for the test eval, `run_experiments.py:420`). So a failed API call, such as an HTTP 429 rate-limit error, is scored as a failed example instead of stopping the run. OpenAI lists 5,000 RPM / 2M TPM for `gpt-4.1-mini` on its lowest tier ("Build"). 32 threads with a 16k `max_tokens` each may exceed that if requested output tokens count toward TPM.
  → Before Table 2, run a ~$1 pilot (step 4.7: Baseline on one benchmark) and check `run_log.txt` for rate-limit errors. If there are any, lower `--num_threads` (this only affects speed) or enable retries as a documented, opt-in deviation.
  → *Check (B):* seen directly with a dummy key: failed calls were scored as zero, and the run carried on. A crashed or unreachable Arbor server has the same effect, so grep `run_log.txt` for errors after every pilot and run.

### D. Orchestration and data pitfalls

- **F18. Dry runs and pilots collide with real runs.** The run directory and cache directory depend only on benchmark, program, optimizer, LM and seed; `--dry_run` doesn't change them.
  - A finished dry run leaves `evaluation_results/`, which marks the real run complete, so it gets **skipped** (`run_experiments.py:203`).
  - A GEPA run started without `--use_cache_from_opt` creates its own cache directory. The real run later fails its `cache_from_source_run.txt` assertion.
  → Every dry run and pilot goes to a separate `BASE_EXPERIMENT_DIR` (step 3.1). The exceptions are pilots 4.3 and 4.5 (the Baselines and the first MIPROv2), and the API Baselines in 4.7. Those are real runs and go to the real directory.
- **F19. The authors' archive must not go into `BASE_EXPERIMENT_DIR`.** Extracting it into `experiment_runs_data/` has the same effect as F18: all runs are skipped, or start from the authors' caches.
  → Extract into `reference_runs_data/` (step 2.2).
- **F20. Runs wait on each other.** MIPROv2 waits for Baseline, and GEPA, GEPA-MERGE and the ablation wait for MIPROv2. Each waiting run polls every 100 s, forever, until the source run has `evaluation_results/` (`run_experiments.py:118-119`), then copies its DSPy cache. The wait is triggered by the `--use_cache_from_opt` CLI flag, which the generated commands include.
  → A job submitted too early burns GPU hours while idle. Chain jobs with SLURM dependencies: Baseline → MIPROv2 → GEPA variants.
- **F21. `--dry_run` doesn't shrink the budget.** It cuts the data to 2 examples, but GEPA still gets `max_metric_calls` from the hard-coded `known_max_calls` (6,871 for HotpotQA; `experiment_configs.py:188`, `run_experiments.py:354`). MIPROv2-Heavy keeps its heavy settings.
  → Smoke tests need a cap (step 3.1).
- **F22. Launch commands.** The generator emits 72 commands (6 benchmarks × 1 program × 6 optimizers × 2 LMs, seed 0), and it downloads all datasets while building the benchmarks. The Qwen commands contain `"api_base": "http://localhost:{portnum}/v1/"` literally: the substitution at `run_experiments.py:247` is unreachable because `launch_arbor` is forced to `False`.
  → Filter the commands and fill in the port yourself (step 3.4).
- **F23. BM25 artifacts land in the source tree.** The first HoVer or HotpotQA search downloads `wiki.abstracts.2017.tar.gz` into the current directory and writes the corpus and index into `gepa_artifact/benchmarks/hover/` (`hover_program.py:41`). Only `retriever_cache/` (`hover_program.py:89`) is gitignored. The init lock only guards threads, not processes.
  → Gitignore these files (step 0.2) and build the index once before parallel jobs (step 1.7). The shared `retriever_cache` is harmless, since BM25 is deterministic.
- **F24. Data splits are fixed.** The base class caps the splits at 150 train / 300 val / 300 test, sampled with seed 1 (`gepa_artifact/benchmarks/benchmark.py:30-32,50`). Some benchmarks define smaller splits themselves:
  - PUPA: 111 / 111 / 221 (`papillon_data.py`).
  - AIME: half of `aimo-validation-aime` each for train and val, and the 30 AIME-2025 problems repeated 5× as test, giving 150 (`AIME_data.py`).
  - LiveBench-Math: about ⅓ each (`livebenchmath_data.py`).

  A seed other than 0 reshuffles train and val only; test stays fixed (`run_experiments.py`).

### E. Fidelity of the reproduction

- **F25. Thinking mode was on.** The Qwen3 sampling settings (temperature 0.6, top-p 0.95, top-k 20; `experiment_configs.py:41`) are Qwen's recommended settings for thinking mode. The paper doesn't say whether thinking was enabled.
  → *Check (B):* thinking was on. The planned check couldn't work, because the metric logs store `prediction: null`. The cached raw Qwen responses show the thinking in `reasoning_content`: Arbor starts vLLM without a reasoning parser, and litellm splits the thinking off on the client side.
  → Keep Qwen3's default chat template and `max_tokens=16384`; there's nothing else to configure. Thinking dominates token counts and compute, and pilot 4.3 measures it (F33).
- **F26. Hardware differs from the authors'.** They used 4-GPU tensor-parallel servers per run on 80 GB GPUs; I use 1 H100 shared by several runs. This should affect speed only.
  → Expect statistically similar but not identical results: sampling uses temperature 0.6 (Qwen) or 1.0 (GPT-4.1-mini). The DSPy cache keeps repeated identical calls deterministic within a run.
- **F27. The paper uses one seed per cell** (`SEEDS = [0]`).
  → Differences of a few points may be noise. *Check (B):* the artifact shows a 4.7-point swing between two runs of the same method. So extra seeds for the key comparisons (step 7.4) are the first extension, not an optional item.
  → The paper's own GRPO row in Table 1 prints an aggregate of 48.91, while its cells average 48.97: a small inconsistency in the paper.
- **F28. GRPO setup per the paper.** "1xH100/A100 (80 GB memory) with separate GPUs for inference rollouts": LoRA rank 16, α 64, dropout 0.05, learning rate 1e-5, 500 steps, group size 12, batch 48, so 24,000 rollouts. This matches `arbor_train.yaml` (inference on GPUs 1 and 2, training on GPU 0) and `TRAIN_KWARGS_GRPO_DEFAULT`.
- **F30. Not everything in the paper is in this repo.** These have no implementation here:
  - Table 2: Trace (OptoPrime), TextGrad and MIPROv2-No-Demos.
  - Table 3: BeamSearch.
  - Figures 7–8: NPU/CUDA kernel experiments with GPT-4o.

  GEPA-Qwen-Opt (Table 2) means the Qwen3-8B-optimized GEPA prompts evaluated on GPT-4.1-mini without changes. It needs a small extra evaluation script.
  → The reproducible scope is Tables 1–3 minus these rows. State this in the write-up.

### F. From B's check against the reference runs (2026-11-04)

- **F33. Qwen token counts aren't in the artifact, and our runs won't record them either.** `evaluation_result.txt` shows 0 tokens for every Arbor run, and the cached responses carry no usage.
  - The cause: Arbor's server (`arbor/server/services/inference/vllm_serve.py` in the fork) returns no `usage` field.
  - `calculate_stats` sums `usage` from the LM history (`scripts/run_experiments.py:47-48`), so it finds nothing.
  → The token table from step 2.3 exists only for GPT-4.1-mini. Pilot 4.3 has to count Qwen tokens itself: from Arbor's response log (F9), with the Qwen3 tokenizer (step 3.5).
- **F34. Target numbers against the shipped runs.** 46 of 58 cells match, 1 differs, and 11 have no run.
  - The one that differs: the paper's GPT-4.1-mini LiveBench-Math GEPA cell (64.13) is the GEPA+Merge run. The shipped GEPA run scores 59.43.
  - There are no reference runs at all for the Qwen3-8B math cells.
  → B's AIME-2025 and LiveBench-Math results can only be compared with the paper's numbers, not with the authors' logs (see the notes in §4).
- **F35. The appendix's reflection-call counts match only the GEPA+Merge runs.** Plain GEPA on AIME made 100 reflection calls; the appendix table says 24.
  → Count them in our own runs from `instruction_proposer_inpouts.jsonl` in each run directory. The typo in the name is in the code itself (`gepa_artifact/gepa/gepa.py:549`).
- **F36. GEPA-family runs overshoot their budget.** Every shipped GEPA-family run used 2–23% more metric calls than `max_metric_calls`, because the last validation pass isn't cut off.
  → Ours will do the same. Report both the budget and the calls actually used, and add up to ~20% to GEPA's SBU estimate (§2, step 5.1).
- **F37. The authors' run times, from their logs.**
  - Qwen3-8B on 4-GPU servers: GEPA 1.2–6.4 h, MIPROv2 1.1–4.7 h. GRPO on 3 GPUs: 12–24 h.
  - GPT-4.1-mini runs: 1–4 h, except GEPA on AIME at 9 h.
  → §2's ~15 h for GEPA on HotpotQA on one H100 is on the safe side, and the GRPO estimate matches. MIPROv2 keeps the GPU busy, so on one GPU it may take up to 2–3× longer than §2 assumes; pilot 4.5 settles this.
- **F38. Baselines can be replayed for free.** The shipped DSPy caches don't depend on the API key. So a reference Baseline can be replayed on Snellius without a GPU or a key (`scripts/replay_test_eval.py`, on B's `feat/replay-and-analysis`).
  - Optimized candidates still miss the cache, for an unknown reason.
  - A final GEPA test evaluation through the API costs about $1.
  → Use the replay as a pipeline check (pilot 4.0) before the first GPU job.
- **F39. Search-tree evidence for the write-up.**
  - GEPA keeps about 19 of about 100 proposals.
  - In 64–87% of iterations, the sampled parent isn't the current best candidate. In 8 of 10 GEPA runs, the final prompt descends from such a pick.
  - Picking the best candidate per validation example would add 7–27 points over the best single candidate.
  → This supports the paper's case for Pareto-based selection. Check whether our runs show the same pattern (step 8.3).

---

## 4. Paper target numbers

Transcribed from the arXiv HTML version. Check against the PDF before the final comparison.

**Table 1: Qwen3-8B**

| Optimizer | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math | Aggregate |
|---|---|---|---|---|---|---|---|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 27.33 | 48.70 | 45.23 |
| GRPO | 43.33 | 35.88 | 38.67 | 86.66 | 38.00 | 51.26 | 48.91 |
| MIPROv2 | 55.33 | 36.22 | 47.33 | 81.55 | 20.00 | 46.60 | 47.84 |
| GEPA | 62.33 | 38.61 | 52.33 | 91.85 | 32.00 | 51.95 | 54.85 |
| GEPA+Merge | 64.33 | 28.23 | 51.67 | 86.26 | 32.00 | 51.95 | 52.40 |
| *Rollout budget, GEPA (+Merge)* | 6871 | 3593 | 7051 | 2426 | 1839 | 1839 | 3936 |
| *Rollout budget, GRPO* | 24000 | 24000 | 24000 | 24000 | 24000 | 24000 | 24000 |

_Note (F34): the artifact has no reference runs for the Qwen3-8B AIME-2025 and LiveBench-Math cells. Our results there can only be compared with these printed numbers._

**Table 2: GPT-4.1-mini** (rows that can be reproduced with this repo)

| Optimizer | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math | Aggregate |
|---|---|---|---|---|---|---|---|
| Baseline | 38.00 | 47.79 | 46.33 | 78.57 | 49.33 | 58.20 | 53.03 |
| MIPROv2 | 58.00 | 49.15 | 48.33 | 83.37 | 51.33 | 61.84 | 58.67 |
| GEPA | 69.00 | 52.72 | 51.67 | 94.47 | 59.33 | 64.13 | 65.22 |
| GEPA+Merge | 65.67 | 55.95 | 56.67 | 96.46 | 59.33 | 64.13 | 66.36 |

_Note (F34): the GEPA LiveBench-Math cell (64.13) matches the shipped GEPA+Merge run. The shipped plain GEPA run scores 59.43._

---

## 5. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-04 | Run everything on Snellius; the Mac is a fallback only | One environment keeps results consistent |
| 2026-11-04 | Start with Qwen3-8B on the 5 OpenAI-free benchmarks; GRPO, PUPA and GPT-4.1-mini are optional | No API keys yet; GRPO is about 5–10× the cost of everything else combined |
| 2026-11-04 | After setup, run pilots (Phase 4) and re-estimate the compute (Phase 5) before any main run | The first-principles estimate is ±2×, and the course budget is shared and expires 2026-12-18 |
| 2026-11-04 | First scope: Table 1 on HotpotQA, HoVer, IFBench, AIME-2025 and LiveBench-Math with Baseline, MIPROv2, GEPA and GEPA+Merge | No OpenAI key yet, so no PUPA. GRPO alone is ~84% of all SBUs |
| 2026-11-04 | Split by benchmark: A has HotpotQA and HoVer; B has IFBench, AIME-2025 and LiveBench-Math. Shared branch `reproduction` | Each benchmark's chain has to stay in one experiment directory (F20); the split gives each of us a similar SBU load |
| 2026-11-04 | Adopt B's check of the plan: corrections to F7, F13, F25, F27 and F29, and new findings F33–F39 | B checked the plan against the authors' logs, caches and code |
| 2026-11-04 | Pin the main environment to the package versions recorded by the original runs (F7), instead of a date cut-off | The exact versions are available, which is closer to the original setup |
| 2026-11-04 | Phase 2 stays done on B's laptop and isn't repeated on Snellius | The reference analysis doesn't depend on where it runs |

## 6. Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, with types `feat`, `fix`, `docs`, `chore`, `build`, `refactor`, `test`.
- Upstream defaults stay unchanged. Every addition (cap, directory override, job scripts) is opt-in, so the original experiment matrix can still be run as is.
- Secrets live outside the repo (`~/.config/gepa/secrets.env`).

### Team workflow

- **Branches:**
  - `main` stays equal to upstream; update it only with `git pull upstream main`.
  - `reproduction` is the shared branch for the plan, the tooling and the results.
  - Code changes go through short branches (`feat/...`, `fix/...`) and a pull request into `reproduction`, reviewed by the other person.
  - Results (§7) are committed straight to `reproduction` after `git pull --rebase`, as `docs(results): ...`.
- **Who runs which benchmark:** a benchmark's whole chain (Baseline → MIPROv2 → GEPA variants) has to run in one person's experiment directory, because each step copies the previous step's cache (F20).

  | Person | Benchmarks | Estimated SBU (§2, ±2×) |
  |---|---|---|
  | A | HotpotQA, HoVer | ~8.3k |
  | B | IFBench, AIME-2025, LiveBench-Math | ~6.3k |
- **Same environment:** both use the committed `uv.lock` (`uv sync --locked`), and both build the Arbor environment with `--exclude-newer 2025-08-04`. Every row in §7.2 records the commit it ran on (`git rev-parse --short HEAD`).
- **Shared budget:** we both draw from `gpuuva090`. Check `budget-overview` before submitting, and log each job's SBU in §7.2.
- **Raw data:** run directories stay in each person's home.
  - For the final figures, B packs their `experiment_runs/` into a tarball under `/scratch-shared/<user>/`, readable by group `gpuuva090_access`. Note that this group includes everyone on the course account.
  - A unpacks it next to their own runs.
- **No merge conflicts in §7:** each of us edits only their own cells and rows, and adds to §7.2 at the bottom instead of reordering.

---

## 7. Results

Scores are the `score` column of `evaluation_results/evaluation_result.txt`: test set, in %. Fill in "ours"; the paper's value is in brackets. The letter after each benchmark says whose it is.

### 7.1 Table 1: Qwen3-8B (without PUPA and GRPO)

| Variant | HotpotQA (A) | HoVer (A) | IFBench (B) | AIME-2025 (B) | LiveBench-Math (B) |
|---|---|---|---|---|---|
| Baseline | – (42.33) | – (35.33) | – (36.90) | – (27.33) | – (48.70) |
| MIPROv2 | – (55.33) | – (47.33) | – (36.22) | – (20.00) | – (46.60) |
| GEPA | – (62.33) | – (52.33) | – (38.61) | – (32.00) | – (51.95) |
| GEPA+Merge | – (64.33) | – (51.67) | – (28.23) | – (32.00) | – (51.95) |

### 7.2 Run log (one row per job, add new rows at the bottom)

| Date | Who | Job ID | Benchmark | Variant | Commit | Wall-clock | SBU | Score | Notes |
|---|---|---|---|---|---|---|---|---|---|

### 7.3 Pilot measurements

| Step | Who | Quantity | Measured | Assumed in §2 |
|---|---|---|---|---|
| 2.3 | B | Thinking mode in the authors' runs | on (cached `reasoning_content`, F25) | on |
| 2.3 | B | Authors' run times on 4-GPU servers: GEPA / MIPROv2 / GRPO (3 GPUs) | 1.2–6.4 h / 1.1–4.7 h / 12–24 h (F37) | ≤15 h / ≤4.1 h / 12–32 h on our layout |
| 4.2 | B | Server startup (job start until Arbor answers) | – | ~0.15 h |
| 4.3 | A, B | Output tokens per call: HotpotQA / HoVer / IFBench / AIME / LiveBench | – | 700 / 700 / 1,200 / 5,000 / 5,000 |
| 4.3 | A, B | Seconds per rollout at 32 threads: multi-hop / IFBench / math | – | ~1.2 / ~1.1 / ~2.2 |
| 4.4 | A | Seconds per GEPA iteration with 1 / 3 / 6 runs sharing a server | – | ~70 / ~80 / ~90 |
| 4.4 | A | Mean GPU utilization with 1 / 3 / 6 runs | – | – |
| 4.5 | B | MIPROv2 wall-clock on IFBench | – | ~2.2 h |

---

## 8. Analysis of the reference runs (B, 2026-11-04)

B ran this on a laptop, on the extracted reference data. The scripts are in `analysis/` on `feat/replay-and-analysis`; once that branch is merged, the same scripts serve step 8.2 for our own runs.

| What | Result | Used in |
|---|---|---|
| Scripts in `analysis/` | Tables, budget curves, search trees, prompt sizes, generalization gap | 8.1, 8.2 |
| Target numbers vs shipped runs | 46 of 58 cells match, 1 differs (GPT LiveBench-Math GEPA), 11 have no run | F34, §4 |
| Thinking mode | On (cached `reasoning_content`) | F25 |
| Package versions of the original runs | Python 3.12.9, dspy 2.6.23, litellm 1.74.0.post1, openai 1.93.2, pydantic 2.11.7, ujson 5.10.0, cloudpickle 3.1.1, datasets 3.6.0 | F7, step 1.3 |
| Run times | Qwen: GEPA 1.2–6.4 h and MIPROv2 1.1–4.7 h on 4 GPUs; GRPO 12–24 h on 3 GPUs. GPT runs 1–4 h (AIME GEPA 9 h) | F37, §2.2 |
| GPT-4.1-mini cost | All shipped GPT runs ≈ $296 by the artifact's own accounting (≈ €260, more with VAT); one GEPA run $6–25 | §2.3 |
| Budget overshoot | GEPA-family runs use 2–23% more metric calls than `max_metric_calls` | F36 |
| Reflection calls | The appendix table matches only the GEPA+Merge runs (plain GEPA on AIME: 100 calls, not 24) | F35 |
| Search trees | ~19 of ~100 proposals kept; the parent isn't the current best in 64–87% of iterations; 8 of 10 final prompts descend from such a pick; per-example best adds 7–27 points | F39 |
| Seed noise | 4.7-point swing between two runs of the same method | F27, step 7.4 |
