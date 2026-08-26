# GEPA (Code Artifact)

> If you are looking to use GEPA in your own projects, please visit the main repository at [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa). This codebase is intended as a research artifact solely for experiment reproduction.

This repository contains the code and data artifact for the paper titled 'GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning' ([https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)).

**Note:**  

Additional GEPA-SGD and offline baseline work on this branch is summarized in
[`HANDOFF.md`](HANDOFF.md).

---

## Table of Contents

- [Setup](#setup)
- [Obtaining Experiment Data](#obtaining-experiment-data)
- [Configuring Experiments](#configuring-experiments)
- [Generating Experiment Commands](#generating-experiment-commands)
- [Executing the Commands](#executing-the-commands)
- [Visualizing Results](#visualizing-results)
- [Citation](#citation)

---

## Setup

First, make sure you have [`uv`](https://github.com/astral-sh/uv) installed:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, clone the [DSPy](https://github.com/stanfordnlp/dspy) and [Arbor](https://github.com/Ziems/arbor) dependencies. We use slightly modified forks provided via the setup script:

```sh
bash setup_gepa_repo.sh
```

#### For GRPO Experiments Only

If you plan to run GRPO experiments, you must swap the default project configuration:

```sh
mv pyproject_grpo.toml pyproject.toml
uv sync --no-install-package flash-attn
```

Finally, sync your Python environment:

```sh
uv sync
```

---

## Obtaining Experiment Data

We provide the data artifacts (e.g., prompts, experiment logs, etc.) as [experiment_runs_data.tar.gz](experiment_runs_data.tar.gz). Ensure that you have git-lfs installed to obtain the file while cloning.

To extract them, run:

```sh
tar -xvf experiment_runs_data.tar.gz
```

---

## Configuring Experiments

All experiment configurations reside in [`scripts/experiment_configs.py`](scripts/experiment_configs.py).  
The following parameters may be modified:

1. `BASE_EXPERIMENT_DIR`: Directory containing the extracted data artifacts
2. `LM_CONFIGS`: Language model configurations to use
3. `get_benchmarks`: List of benchmarks to run
4. `get_optimizers`: List of optimizer configurations (GEPA, MIPRO, GRPO, ablations, etc.)

---

## Generating Experiment Commands

After configuring experiments, generate executable commands to run each experiment via:

```sh
uv run python -m scripts.generate_launch_commands > launch_commands
```

This will create a `launch_commands` file containing all shell commands needed to launch each experiment.

---

## Executing the Commands

Commands must be executed in an environment where both `OPENAI_API_KEY` and `WANDB_API_KEY` are set.

- [Arbor](https://github.com/Ziems/arbor) is used for local inference and GRPO training.
- For GPU setups, refer to the `.yaml` files in `gepa_artifact/utils/arbor`.

---

## Visualizing Results

Once all experiment commands have finished (or if you use the provided data artifacts, simply after extracting them), result logs will appear under `experiment_runs_data/`.

To reproduce all figures from the GEPA paper, run the notebook [scripts/generate_figures.ipynb](scripts/generate_figures.ipynb).

## Notes
* Please note that the codebase reads whatever OPENAI_API_KEY is active in the environment. Kindly ensure you set the correct OPENAI_API_KEY environment variable. PAPILLON uses an LLM-as-a-judge metric which uses OPENAI_API_KEY irrespective of the task LM you use (this is to ensure consistent judging across runs.). Thanks @jlesner for highlighting this!

---

## Citation

If you use this artifact, please cite:

```bibtex
@misc{agrawal2025gepareflectivepromptevolution,
  title     = {GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author    = {Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
  year      = {2025},
  eprint    = {2507.19457},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url       = {https://arxiv.org/abs/2507.19457},
}
```
