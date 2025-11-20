import argparse
import os
import os
import time
import json
import traceback
import random

from gepa_artifact.utils.capture_stream_logger import Logger
from .experiment_configs import BASE_EXPERIMENT_DIR, get_benchmarks, get_optimizers, get_max_invocations

def write_evaluation_result_to_path(evaluation_result, file_path):
    os.makedirs(file_path, exist_ok=True)
    file_name = f"evaluation_result"
    if evaluation_result.optimizer:
        optimizer_header = "optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens"
        optimizer_values = (
            f"{evaluation_result.optimizer},{evaluation_result.optimizer_cost},"
            f"{evaluation_result.optimizer_input_tokens},{evaluation_result.optimizer_output_tokens},"
        )
    else:
        optimizer_header = ""
        optimizer_values = ""
    with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
        f.write(f"score,cost,input_tokens,output_tokens,{optimizer_header}\n")
        f.write(
            f"{evaluation_result.score},{evaluation_result.cost},{evaluation_result.input_tokens},"
            f"{evaluation_result.output_tokens},{optimizer_values}\n"
        )
    if evaluation_result.optimizer:
        evaluation_result.optimized_program.save(
            os.path.join(file_path, f"optimized_program"),
            save_program=True
        )
    if evaluation_result.optimizer_program_scores:
        with open(
            os.path.join(file_path, f"{file_name}_optimizer_score.txt"), "w"
        ) as f:
            f.write(",".join(evaluation_result.optimizer_program_scores))

def calculate_stats(lm) -> tuple[float, int, int]:
    cost = 0
    input_tokens = 0
    output_tokens = 0
    for i, trace in enumerate(lm.history):
        cost += trace.get("cost", None) or 0
        input_tokens += trace.get("usage", 0).get("prompt_tokens", 0)
        output_tokens += trace.get("usage", 0).get("completion_tokens", 0)

    return cost, input_tokens, output_tokens

def create_lm(lm_config):
    import dspy
    config = lm_config.copy()
    config['model'] = config.pop("new_model_name", config['model'])
    from dspy.clients.lm_local_arbor import ArborProvider
    provider = ArborProvider() if "openai/arbor" in config['model'] else None
    fixed_config = {
        "max_tokens": 16384,  # overriding the dspy defaults
        "num_retries": 0,
        "provider": provider,
    }
    config = {k:v for k, v in config.items() if k != "name"}
    return dspy.LM(**config, **fixed_config)


def get_free_port() -> int:
    """
    Return a randomly selected free TCP port on localhost from a selection of 3-4 ports.
    """
    import random
    import socket
    ports = []
    for _ in range(random.randint(5, 10)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 0))
                ports.append(s.getsockname()[1])
        except Exception as e:
            print(f"Error binding to port: {e}")
    return random.choice(ports)

def run_experiment_and_write_results_actual(
    bm_idx,
    benchmark_name,
    num_threads,
    program_idx,
    prog_name,
    opt_idx,
    optim_name,
    lm_config,
    dry_run=False,
    use_cache_from_opt=None,
    seed=0,
    arbor_config_path=None,
):
    base_experiment_dir = BASE_EXPERIMENT_DIR
    lm_name = lm_config["name"]
    print(f"Running {benchmark_name} with {prog_name} and {optim_name} on {lm_name}")
    runs_dir_basepath = os.path.join(base_experiment_dir, "experiment_runs", f"seed_{seed}")
    run_name = f"{benchmark_name}_{prog_name}_{optim_name}_{lm_name}"
    runs_dir = os.path.join(runs_dir_basepath, run_name)

    #######################
    # Cache Setup:
    # use_cache_from_opt is used to ensure consistency
    #######################
    cache_dir = os.path.join(base_experiment_dir, "experiment_cache_dirs", f"seed_{seed}", run_name)
    if use_cache_from_opt is None:
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_source_run_name = f"{benchmark_name}_{prog_name}_{use_cache_from_opt}_{lm_name}"
        cache_source_cache_subdir = os.path.join("experiment_cache_dirs", f"seed_{seed}", cache_source_run_name)
        cache_source_cache_dir = os.path.join(base_experiment_dir, "experiment_cache_dirs", f"seed_{seed}", cache_source_run_name)
        if not os.path.exists(cache_dir):
            cache_source_run_dir = os.path.join(base_experiment_dir, "experiment_runs", f"seed_{seed}", cache_source_run_name)
            # Now, we will wait indefinitely till the source run has evaluation results ready
            print(f"Waiting for {cache_source_run_dir} to have evaluation results ready...")
            while not os.path.exists(os.path.join(cache_source_run_dir, "evaluation_results")):
                time.sleep(100)
            print(f"Found evaluation results for {cache_source_run_name} in {cache_source_run_dir}.")
            print(f"Copying cache from {cache_source_cache_dir} to {cache_dir}...")
            import shutil
            shutil.copytree(cache_source_cache_dir, cache_dir)
            # Write a small token marker file to indicate this was copied from the source run
            with open(os.path.join(cache_dir, "cache_from_source_run.txt"), "w") as f:
                f.write(f"Copied from {cache_source_cache_dir}")
            print(f"Copied cache from {cache_source_cache_dir} to {cache_dir}. Continuing the current run with the cache from {cache_source_run_name}...")
            assert os.path.exists(os.path.join(cache_dir, ".dspy_cache"))
        else:
            assert os.path.exists(os.path.join(cache_dir, "cache_from_source_run.txt"))
            with open(os.path.join(cache_dir, "cache_from_source_run.txt"), "r") as f:
                assert cache_source_cache_subdir in f.read() # == f"Copied from {cache_source_cache_dir}"

    dspy_cachedir = os.path.join(cache_dir, ".dspy_cache")
    os.environ["DSPY_CACHEDIR"] = dspy_cachedir
    os.environ["DSP_CACHEDIR"] = dspy_cachedir
    os.environ["DSPY_NOTEBOOK_CACHEDIR"] = dspy_cachedir
    os.environ["DSP_NOTEBOOK_CACHEDIR"] = dspy_cachedir
    import dspy

    from gepa_artifact.benchmarks.benchmark import EvaluationResult
    from gepa_artifact.utils.metric_logger import MetricWithLogger, CounterWithLock
    from gepa_artifact.utils.json_default_encoder import json_encoder

    metric_lm_name = lm_config.get("metric_lm_name", lm_config["name"])
    metric_lm = lm_config.get("model", None)

    adapter = dspy.settings.adapter # if "qwen" not in lm_name else XMLAdapter()
    evalsetname = "testset"

    #######################
    # Obtain the benchmark, program and optimizer to execute
    #######################
    benchmark_metas, optimizers = get_benchmarks(), get_optimizers()
    benchmark_meta = benchmark_metas[bm_idx]
    program = benchmark_meta.program[program_idx]
    optimizer_config = optimizers[opt_idx][1]
    benchmark = benchmark_meta.benchmark()

    optimizer_config.langProBe_configs['launch_arbor'] = False

    if use_cache_from_opt is not None:
        assert "use_cache_from_opt" in optimizer_config.langProBe_configs
        assert optimizer_config.langProBe_configs["use_cache_from_opt"] == use_cache_from_opt

    if seed != 0:
        # Shuffle the examples
        print("Shuffling the data splits: train and val")
        train_size = len(benchmark.train_set)
        combined_train_val = benchmark.train_set + benchmark.val_set
        random.Random(seed).shuffle(combined_train_val)
        benchmark.train_set = combined_train_val[:train_size]
        benchmark.val_set = combined_train_val[train_size:]

    assert benchmark_name == (benchmark_meta.name or benchmark.__class__.__name__)
    assert num_threads <= (benchmark_meta.num_threads or os.cpu_count())
    assert prog_name == getattr(program, "_name", program.__class__.__name__)
    assert optim_name == optimizers[opt_idx][0]
    if optimizers[opt_idx][1] is not None:
        assert optimizers[opt_idx][1].name == optim_name

    if optimizer_config is not None and "run_constraints" in optimizer_config.langProBe_configs:
        run_constraints = optimizer_config.langProBe_configs["run_constraints"]
        if "benchmark_name" in run_constraints and benchmark_name not in run_constraints["benchmark_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return

    if getattr(program, "run_constraints", None) is not None:
        run_constraints = program.run_constraints
        if "benchmark_name" in run_constraints and benchmark_name not in run_constraints["benchmark_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return
        if "model_name" in run_constraints and lm_name not in run_constraints["model_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return
        if "optimizer_name" in run_constraints and optim_name not in run_constraints["optimizer_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return

    #######################
    # Check if this experiment has already executed successfully, if so, skip
    #######################
    if os.path.exists(runs_dir) and not os.path.exists(os.path.join(runs_dir, "evaluation_results")):
        # Move the existing directory to a backup location with a timestamp
        import shutil
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{runs_dir}_backup_{timestamp}"
        print(f"Run directory {runs_dir} already exists. Moving to {backup_dir}...")
        try:
            os.rename(runs_dir, backup_dir)
            print(f"Fast-moved {runs_dir} to {backup_dir} via rename.")
        except OSError as e:
            shutil.move(runs_dir, backup_dir)

        directory_existed = False
    else:
        directory_existed = os.path.exists(runs_dir) and os.path.exists(os.path.join(runs_dir, "evaluation_results"))
    
    os.makedirs(runs_dir, exist_ok=True)

    print("Running", benchmark_name, prog_name, optim_name, lm_name, evalsetname, "seed", seed)

    try:
        if (optimizer_config is not None and "launch_arbor" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["launch_arbor"]) or "{portnum}" in lm_config["api_base"]:
            from gepa_artifact.utils.arbor_runner import ArborRunner
            if arbor_config_path:
                arbor_config_file_path = arbor_config_path
            elif "GRPO" in optim_name:
                arbor_config_file_path = os.path.join(os.getcwd(), "utils/arbor/arbor_train.yaml")
                num_gpus = 3
            else:
                import torch
                num_gpus = torch.cuda.device_count()
                if num_gpus == 4:
                    arbor_config_file_path = os.path.join(os.getcwd(), "utils/arbor/arbor_inference.yaml")
                elif num_gpus == 2:
                    arbor_config_file_path = os.path.join(os.getcwd(), "utils/arbor/arbor_inference_2_gpus.yaml")
                else:
                    raise ValueError(f"Number of GPUs {num_gpus} not supported")
            
            if not arbor_config_path and "GRPO" not in optim_name:
                num_gpus = torch.cuda.device_count()
            elif arbor_config_path:
                # If custom path provided, determine num_gpus from torch
                import torch
                num_gpus = torch.cuda.device_count()

            arbor_config = {"config_filepath": arbor_config_file_path, "gpus": list(range(num_gpus))}

            portnum = get_free_port()
            arbor_config["portnum"] = portnum
            arbor_runner_context = ArborRunner(arbor_config["config_filepath"], arbor_config["portnum"], runs_dir)
            arbor_runner_context.__enter__()

            assert "{portnum}" in lm_config["api_base"]
            lm_config["api_base"] = lm_config["api_base"].format(portnum=arbor_config["portnum"])
            print("UPDATED PORT")

        metric_counter = CounterWithLock()

        print(f"Benchmark {benchmark_name} contains {len(benchmark.train_set)} train examples, {len(benchmark.val_set)} val examples and {len(benchmark.test_set)} test examples.")

        if dry_run:
            benchmark.train_set = benchmark.train_set[:2]
            benchmark.val_set = benchmark.val_set[:2]
            benchmark.dev_set = benchmark.dev_set[:2]
            benchmark.test_set = benchmark.test_set[:2]
            print(f"Dry run: only using 2 examples from each set.")

        final_eval_set = benchmark.test_set

        with MetricWithLogger(
            metric_fn=benchmark_meta.metric,
            run_dir=runs_dir,
            counter_with_lock=metric_counter,
            train_dataset=benchmark.train_set,
            val_dataset=benchmark.val_set,
            test_dataset=benchmark.test_set,
            # log_example=True,
            log_prediction=True
        ) as metric_fn_with_logger, Logger(os.path.join(runs_dir, "run_log.txt")) as logger: # 
            # logger = Logger(os.path.join(runs_dir, "run_log.txt"))
            if optimizer_config is not None and "launch_arbor" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["launch_arbor"]:
                logger.log("Arbor in session:", arbor_runner_context.session_name)

            #######################
            # For GEPA, if feedback_fn_maps is not provided, we create a default feedback function based on metric_with_feedback
            # and apply it to all predictors in the program.
            #######################
            if "GEPA" in optim_name:
                
                if benchmark_meta.feedback_fn_maps is None or benchmark_meta.feedback_fn_maps[program_idx] is None:
                    def feedback_func(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
                        pred = benchmark_meta.metric_with_feedback(module_inputs, module_outputs, None)
                        return {
                            "feedback_score": pred.score,
                            "feedback_text": pred.feedback,
                        }

                    feedback_fn_map = {k:feedback_func for k, v in program.named_predictors()}
                else:
                    feedback_fn_map = benchmark_meta.feedback_fn_maps[program_idx]
                    assert all(k in feedback_fn_map for k, _ in program.named_predictors())

                optimizer_config.init_args.update({
                    "named_predictor_to_feedback_fn_map": feedback_fn_map,
                    "knowledgebase_qe": None,
                    "logger": logger,
                    "run_dir": runs_dir,
                    "use_wandb": True,
                    "wandb_api_key": wandb_api_key,
                })

                logger.log("Optimizer config:", optimizer_config)

            if not os.path.exists(os.path.join(runs_dir, "config.json")):
                with open(os.path.join(runs_dir, "config.json"), "w") as f:
                    json.dump({
                        "benchmark_name": benchmark_name,
                        "program_name": prog_name,
                        "program": program,
                        "optimizer_name": optim_name,
                        "lm_name": lm_name,
                        "lm_config": lm_config,
                        "num_threads": num_threads,
                        "optimizer_config": optimizer_config,
                        "metric_lm_name": metric_lm_name,
                        "metric_lm": metric_lm,
                    }, f, default=json_encoder)

            if directory_existed:
                print(f"Run directory {runs_dir} already exists. Skipping...")
                return

            eval_results = EvaluationResult(
                benchmark=benchmark_name,
                program=prog_name,
            )

            if optim_name == "Baseline" or optimizer_config is None:
                # Only run the final evaluation
                optimized_program = program
                eval_results.optimized_program = optimized_program
            else:
                # Run the optimizer, and then run the final evaluation
                optimizer = optimizer_config.optimizer
                init_args = optimizer_config.init_args
                
                #######################
                # Add various configurations to the init_args of the optimizer
                #######################
                if num_threads and "num_threads" in init_args:
                    init_args["num_threads"] = num_threads
                if "provide_logdir_in_init" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["provide_logdir_in_init"]:
                    init_args["log_dir"] = os.path.join(runs_dir, "optimizer_logs")
                    os.makedirs(init_args["log_dir"], exist_ok=True)
                
                if "add_max_errors_to_initargs" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["add_max_errors_to_initargs"]:
                    init_args["max_errors"] = (len(benchmark.train_set) + len(benchmark.val_set)) * 100

                if "add_max_metric_calls" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["add_max_metric_calls"]:
                    # f"{benchmark_name}_{prog_name}_{optim_name}_{lm_name}"
                    num_mipro_invocations = get_max_invocations(benchmark_name, prog_name, metric_lm_name, opt=optimizer_config.langProBe_configs.get("max_metric_calls_source_opt_name"))
                    assert num_mipro_invocations is not None, f"Could not find max invocations for {benchmark_name}, {prog_name}, {metric_lm_name}"
                    init_args["max_metric_calls"] = num_mipro_invocations
                
                if "add_wandb_configs_to_initargs" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["add_wandb_configs_to_initargs"]:
                    init_args["use_wandb"] = True
                    init_args["wandb_api_key"] = wandb_api_key
                    init_args["wandb_run_name"] = run_name + "_seed_" + str(seed)
                    init_args["wandb_project_name"] = "GEPA"

                if "exclude_seed_from_initargs" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["exclude_seed_from_initargs"]:
                    init_args.pop("seed", None)
                else:
                    init_args['seed'] = seed

                compile_args = optimizer_config.compile_args
                langProBe_configs = optimizer_config.langProBe_configs | {"name": optimizer_config.name}
                optimizer = optimizer(metric=metric_fn_with_logger, **init_args)

                lm_for_optimizer = create_lm(lm_config)
                dspy.configure(lm=lm_for_optimizer, adapter=adapter)

                if "set_lm_before_optimizer" in langProBe_configs and langProBe_configs["set_lm_before_optimizer"]:
                    program.set_lm(lm_for_optimizer)
                
                print("STARTING COMPILATION FOR", benchmark_name, prog_name, optim_name, lm_name, evalsetname, "seed", seed)

                if "add_valset_to_trainset" in langProBe_configs and langProBe_configs["add_valset_to_trainset"]:
                    assert "use_valset" not in langProBe_configs or not langProBe_configs["use_valset"]
                    optimized_program = optimizer.compile(
                        program,
                        trainset = benchmark.train_set + benchmark.val_set,
                        **compile_args,
                    )
                elif "use_valset" in langProBe_configs and langProBe_configs["use_valset"]:
                    optimized_program = optimizer.compile(
                        program,
                        trainset=benchmark.train_set,
                        valset=benchmark.val_set,
                        **compile_args,
                    )
                else:
                    assert False

                if "use_model_name_from_optimized_program" in langProBe_configs and langProBe_configs["use_model_name_from_optimized_program"]:
                    lm_config["new_model_name"] = lm_for_optimizer.model
                    with open(os.path.join(runs_dir, "lm_config.json"), "w") as f:
                        json.dump(lm_config, f, default=json_encoder)

                (
                    eval_results.optimizer_cost,
                    eval_results.optimizer_input_tokens,
                    eval_results.optimizer_output_tokens,
                ) = calculate_stats(lm_for_optimizer)

                eval_results.optimizer = optim_name
                eval_results.optimized_program = optimized_program

                dspy.configure(lm=None, adapter=None)
                del lm_for_optimizer

            evaluate_prog = dspy.Evaluate(
                devset=final_eval_set,
                metric=metric_fn_with_logger,
                num_threads=num_threads,
                display_progress=True,
                max_errors=len(final_eval_set)*10,
                provide_traceback=True,
            )

            eval_lm = create_lm(lm_config)
            dspy.configure(lm=eval_lm, adapter=adapter)
            score = evaluate_prog(optimized_program)
            eval_results.score = score
            eval_results.cost, eval_results.input_tokens, eval_results.output_tokens = calculate_stats(
                eval_lm
            )

            dspy.configure(lm=None, adapter=None)
            del eval_lm

            write_evaluation_result_to_path(
                eval_results,
                os.path.join(runs_dir, "evaluation_results"),
            )
    finally:
        if optimizer_config is not None and "launch_arbor" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["launch_arbor"]:
            arbor_runner_context.__exit__(None, None, None)

def run_experiment_and_write_results(*args, **kwargs):
    try:
        return run_experiment_and_write_results_actual(*args, **kwargs)
    except Exception as e:
        print(traceback.format_exc())
        raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description='A program with boolean arguments.')
    
    # Argument 1: dry_run
    # Defaults to False. If --dry_run is passed, it becomes True.
    parser.add_argument(
        '--dry_run',
        action='store_true',
        default=False,
        help='Set to true for a dry run (default: False)'
    )

    parser.add_argument('--bm_idx', type=int, required=True, help='Index of the benchmark to run')
    parser.add_argument('--benchmark_name', type=str, required=True, help='Name of the benchmark to run')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use for the benchmark (default: 1)')
    parser.add_argument('--program_idx', type=int, required=True, help='Index of the program to run')
    parser.add_argument('--prog_name', type=str, required=True, help='Name of the program to run')
    parser.add_argument('--opt_idx', type=int, required=True, help='Index of the optimizer to run')
    parser.add_argument('--optim_name', type=str, required=True, help='Name of the optimizer to run')
    parser.add_argument('--lm_config', type=json.loads, required=True, help='JSON string of the LM configuration')
    parser.add_argument('--use_cache_from_opt', type=str, default=None, help='Name of the optimizer to use cache from (default: None)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--arbor_config_path', type=str, default=None, help='Path to arbor config YAML file (default: None, uses automatic selection)')

    args = parser.parse_args()

    if 'api_key' in args.lm_config and args.lm_config['api_key'].startswith('env:'):
        # If the API key is specified as an environment variable, we will set it here
        env_var = args.lm_config['api_key'].split(':')[1]
        if env_var in os.environ:
            args.lm_config['api_key'] = os.environ[env_var]
        else:
            raise ValueError(f"Environment variable {env_var} not found. Please set it before running the script. It is required for the LM configuration.")

    return args

if __name__ == "__main__":
    assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."
    assert "WANDB_API_KEY" in os.environ, "Please set the WANDB_API_KEY environment variable."
    openai_api_key = os.environ["OPENAI_API_KEY"]
    wandb_api_key = os.environ["WANDB_API_KEY"]
    args = parse_arguments()
    run_experiment_and_write_results(
        **args.__dict__,
    )
