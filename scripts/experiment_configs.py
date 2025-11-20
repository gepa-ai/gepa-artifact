from pathlib import Path

BASE_EXPERIMENT_DIR = str((Path(__file__).parent.parent / "experiment_runs_data").resolve())

MAX_CONTEXT_LENGTH = 8192
MAX_CONTEXT_LENGTH_TRAINING = 8192
LAUNCH_KWARGS = {
    "max_context_length": MAX_CONTEXT_LENGTH
}
SAMPLING_TEMPERATURE = 0.6
TRAIN_KWARGS_GRPO_DEFAULT = {
    "update_interval": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 20,
    "temperature": SAMPLING_TEMPERATURE,
    "beta": 0.01,
    "learning_rate": 1e-5,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "max_prompt_length": None,
    "max_completion_length": None,
    "scale_rewards": True,
    "max_grad_norm": 0.1,
    "lora": True,
    'report_to': "wandb",
    'log_completions': True,
    'logging_steps': 100,
    "generation_batch_size": 12,
}
TRAIN_KWARGS_GRPO_QWEN = {**TRAIN_KWARGS_GRPO_DEFAULT}

# Add/modify available LMs here.
LM_CONFIGS = [
    {
        "name": "qwen3-8b",
        "model": "openai/arbor:qwen/qwen3-8b",
        "api_key": "API_KEY",
        "api_base": "http://localhost:{portnum}/v1/",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "launch_kwargs": LAUNCH_KWARGS,
        "train_kwargs": TRAIN_KWARGS_GRPO_QWEN,
    },
    # {
    #     "name": "gpt-41-mini",
    #     "model": "openai/gpt-4.1-mini-2025-04-14",
    #     "api_key": "env:OPENAI_API_KEY",
    #     "temperature": 1.0,
    # },
]

def get_benchmarks():
    from gepa_artifact.benchmarks.hover import benchmark as hover_metas
    from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpotQA_metas
    from gepa_artifact.benchmarks.papillon import benchmark as papillon_metas
    from gepa_artifact.benchmarks.IFBench import benchmark as ifbench_metas
    from gepa_artifact.benchmarks.livebench_math import benchmark as math_metas
    from gepa_artifact.benchmarks.AIME import benchmark as aime_metas

    from gepa_artifact.utils.optimizers import OptimizerConfig

    # benchmark_metas = hover_metas + hotpotQA_metas + papillon_metas + ifbench_metas + math_metas + aime_metas
    benchmark_metas = math_metas + aime_metas
    return benchmark_metas

def get_optimizers():
    import dspy
    from gepa_artifact.gepa.gepa import GEPA
    from dspy.teleprompt.grpo import GRPO
    from gepa_artifact.utils.optimizers import OptimizerConfig
    optimizers = [
        ("Baseline", 
            OptimizerConfig(
                optimizer=None,
                init_args={},
                compile_args={},
                langProBe_configs=dict(
                    launch_arbor=True,
                ),
                name="Baseline",
            )
        ),
        (
            "MIPROv2-Heavy", 
            OptimizerConfig(
                optimizer=dspy.teleprompt.MIPROv2,
                init_args=dict(auto="heavy", max_errors=10000),
                compile_args=dict(
                    requires_permission_to_run=False,
                ),
                langProBe_configs=dict(
                    use_valset=True,
                    save_candidate_score=True,
                    provide_logdir_in_init=True,
                    add_max_errors_to_initargs=True,
                    launch_arbor=True,
                    use_cache_from_opt="Baseline",
                ),
                name="MIPROv2-Heavy",
            )
        ),
        (
            "GEPA-MERGE",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(run_linearized_gepa=False, use_merge=True, set_for_merge_minibatch='val', track_scores_on='val'),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    add_max_metric_calls=True,
                    max_metric_calls_source_opt_name="MIPROv2-Heavy",
                    launch_arbor=True,
                    use_cache_from_opt="MIPROv2-Heavy",
                ),
                name="GEPA-MERGE",
            )
        ),
        (
            "GEPA",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(run_linearized_gepa=False, use_merge=False, set_for_merge_minibatch='val', track_scores_on='val'),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    add_max_metric_calls=True,
                    max_metric_calls_source_opt_name="MIPROv2-Heavy",
                    launch_arbor=True,
                    use_cache_from_opt="MIPROv2-Heavy",
                ),
                name="GEPA",
            )
        ),
        (
            "Abl-SelectBestCandidate",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(run_linearized_gepa=True, use_merge=False, set_for_merge_minibatch='val', track_scores_on='val'),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    add_max_metric_calls=True,
                    max_metric_calls_source_opt_name="MIPROv2-Heavy",
                    launch_arbor=True,
                    use_cache_from_opt="MIPROv2-Heavy",
                ),
                name="Abl-SelectBestCandidate",
            )
        ),
        (
            "GRPO",
            OptimizerConfig(
                optimizer=GRPO,
                init_args=dict(
                    multitask=True,
                    exclude_demos=False,
                    num_train_steps=500,
                    num_threads=25,
                    use_train_as_val=False,
                    num_steps_for_val=20,
                    sampling_temperature=SAMPLING_TEMPERATURE,
                    num_dspy_examples_per_grpo_step=4,
                    num_rollouts_per_grpo_step=12,
                    grpo_group_size=12,
                    report_train_scores=False,
                    variably_invoked_predictor_grouping_mode="fill",
                    variably_invoked_predictor_fill_strategy="randint",
                    max_context_length=MAX_CONTEXT_LENGTH_TRAINING,
                ),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    add_valset_to_trainset=False,
                    use_model_name_from_optimized_program=True,
                    set_lm_before_optimizer=True,
                    launch_arbor=True,
                    add_wandb_configs_to_initargs=True,
                ),
                name="GRPO",
            )
        ),
    ]

    return optimizers

def get_max_invocations(bench, prog, model, opt):
    known_max_calls = {
        ('HotpotQABench', 'HotpotMultiHop', 'MIPROv2-Heavy'): 6871,
        ('Papillon', 'PAPILLON', 'MIPROv2-Heavy'): 2426,
        ('hoverBench', 'HoverMultiHop', 'MIPROv2-Heavy'): 7051,
        ('IFBench', 'IFBenchCoT2StageProgram', 'MIPROv2-Heavy'): 3593,
        ('LiveBenchMathBench', 'CoT', 'MIPROv2-Heavy'): 1839,
        ('AIMEBench', 'CoT', 'MIPROv2-Heavy'): 1839,
    }

    if (bench, prog, opt) in known_max_calls:
        return known_max_calls[(bench, prog, opt)]

    raise Exception(
        f"Could not find max invocations for {bench}, {prog}, {opt}. "
        "Please add it to the known_max_calls dictionary in get_max_invocations."
    )
