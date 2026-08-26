from copy import deepcopy
import os
import traceback
import dspy
import random
import itertools
import json

import dspy.teleprompt
import dspy.teleprompt.teleprompt
import dspy
import wandb

from dspy.teleprompt.bootstrap_finetune import bootstrap_trace_data, FailedPrediction

from .entropy_utils import remove_dominated_programs

class GEPAState:
    program_candidates: list[dspy.Module]
    parent_program_for_candidate: list[list[int | None]]

    program_full_scores: list[float]
    program_full_scores_val_set: list[float]
    
    pareto_front: list[float]
    
    program_at_pareto_front: list[set[int]]
    program_at_pareto_front_valset: list[set[int]]

    prog_candidate_train_subscores: list[list[float]]
    prog_candidate_val_subscores: list[list[float]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals_per_trainval_instance: float
    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    running_linearized_gepa: bool

    rng1: random.Random
    rng2: random.Random

    full_program_trace: list

    per_program_tracked_scores: list[float]
    track_scores_on: str

    def __init__(
        self, 
        base_program: dspy.Module, 
        base_trainset_eval_output: tuple[float, list[dspy.Prediction], list[float]],
        base_valset_eval_output: tuple[float, list[dspy.Prediction], list[float]],
        seed: int, 
        track_scores_on: str,
        run_linearized_gepa: bool=False
    ):
        base_program_full_score = base_trainset_eval_output[0]
        base_pareto_front = base_trainset_eval_output[2]
        valset_base_score = base_valset_eval_output[0]
        base_valset_pareto_front = base_valset_eval_output[2]

        base_program_lm = base_program.get_lm()
        first_program_candidate = base_program.deepcopy()
        first_program_candidate.set_lm(base_program_lm)

        self.program_candidates = [first_program_candidate]
        self.program_full_scores = [base_program_full_score]
        self.program_full_scores_val_set = [valset_base_score]
        if base_program_full_score is not None:
            # track_scores_on == 'train_val'
            self.per_program_tracked_scores = [(base_program_full_score*len(base_pareto_front) + valset_base_score*len(base_valset_pareto_front)) / (len(base_pareto_front) + len(base_valset_pareto_front))]
        else:
            # track_scores_on == 'val'
            self.per_program_tracked_scores = [valset_base_score]

        self.pareto_front = base_pareto_front
        self.pareto_front_valset = base_valset_pareto_front
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front = [{0} for _ in range(len(base_pareto_front))]
        self.program_at_pareto_front_valset = [{0} for _ in range(len(base_valset_pareto_front))]

        self.list_of_named_predictors = [k[0] for k in base_program.named_predictors()]
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1
        self.rng1 = random.Random(seed)
        self.rng2 = random.Random(seed+1)

        self.prog_candidate_train_subscores = [list(base_trainset_eval_output[2])]
        self.prog_candidate_val_subscores = [list(base_valset_eval_output[2])]
        self.num_metric_calls_by_discovery = [0]

        self.running_linearized_gepa = run_linearized_gepa
        self.track_scores_on = track_scores_on

        self.full_program_trace = []

    def is_consistent(self):
        if self.track_scores_on == 'train_val':
            assert len(self.program_candidates) == len(self.program_full_scores)
            assert len(self.prog_candidate_train_subscores) == len(self.program_candidates)
            assert len(self.pareto_front) == len(self.program_at_pareto_front)
            for prog_list in self.program_at_pareto_front:
                for prog_idx in prog_list:
                    assert prog_idx < len(self.program_candidates), "Program index in pareto front exceeds number of program candidates"

        assert len(self.program_candidates) == len(self.program_full_scores_val_set)
        assert len(self.program_candidates) == len(self.per_program_tracked_scores)
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        
        assert len(self.prog_candidate_val_subscores) == len(self.program_candidates)
        assert len(self.pareto_front_valset) == len(self.program_at_pareto_front_valset)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        for prog_list in self.program_at_pareto_front_valset:
            for prog_idx in prog_list:
                assert prog_idx < len(self.program_candidates), "Program index in valset pareto front exceeds number of program candidates"

        return True

    def save(self, run_dir:str):
        for prog_idx, curr_prog in enumerate(self.program_candidates):
            dir_to_save = os.path.join(run_dir, "prog_candidates", str(prog_idx))
            os.makedirs(dir_to_save, exist_ok=True)
            curr_prog_lm = curr_prog.get_lm()
            curr_prog.set_lm(None)
            curr_prog.save(dir_to_save, save_program=True)
            curr_prog.set_lm(curr_prog_lm)
        
        # Save all the other state except programs as pickle
        with open(os.path.join(run_dir, "gepa_state.bin"), 'wb') as f:
            import pickle
            d = {k: v for k, v in self.__dict__.items() if k != 'program_candidates'}
            pickle.dump(d, f)
    
    @staticmethod
    def load(run_dir: str) -> 'GEPAState':
        with open(os.path.join(run_dir, "gepa_state.bin"), 'rb') as f:
            import pickle
            d = pickle.load(f)
        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)
        if not hasattr(state, 'running_linearized_gepa'):
            setattr(state, 'running_linearized_gepa', False)
        state.program_candidates = []
        for i in itertools.count():
            dir_to_load = os.path.join(run_dir, "prog_candidates", str(i))
            if not os.path.exists(dir_to_load):
                break
            prog = dspy.load(dir_to_load)
            state.program_candidates.append(prog)
        
        if state.track_scores_on == 'train_val':
            assert len(state.program_candidates) == len(state.program_full_scores)
            assert len(state.pareto_front) == len(state.program_at_pareto_front)
        else:
            assert len(state.program_candidates) == len(state.program_full_scores_val_set)
            assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)

        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)

        return state

    def update_state_with_new_program(self, parent_program_idx, new_program, trainset_score, trainset_outputs, trainset_subscores, valset_score, valset_outputs, valset_subscores, run_dir, track_scores_on, num_metric_calls_by_discovery_of_new_program):
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)
        # Find the highest predictor id from the parent programs
        max_predictor_id = max([self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx])
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(parent_program_idx)

        if track_scores_on == 'train_val':
            for task_idx, (old_score, new_score) in enumerate(zip(self.pareto_front, trainset_subscores)):
                if new_score > old_score:
                    os.makedirs(os.path.join(run_dir, "generated_best_outputs", f"task_{task_idx}"), exist_ok=True)
                    with open(os.path.join(run_dir, "generated_best_outputs", f"task_{task_idx}", f"iter_{self.i+1}_prog_{new_program_idx}.json"), 'w') as f:
                        json.dump(trainset_outputs[task_idx], f, indent=4, default=json_default)

            self.pareto_front, self.program_at_pareto_front = update_pareto_front(
                new_prog_all_scores=trainset_subscores,
                new_program_idx=new_program_idx,
                pareto_front=self.pareto_front,
                program_at_pareto_front=self.program_at_pareto_front
            )

            self.program_full_scores.append(trainset_score)
            self.prog_candidate_train_subscores.append(trainset_subscores)
            assert len(trainset_subscores) == len(self.program_at_pareto_front)
        
        self.prog_candidate_val_subscores.append(valset_subscores)
        self.program_full_scores_val_set.append(valset_score)
        for task_idx, (old_score, new_score) in enumerate(zip(self.pareto_front_valset, valset_subscores)):
            if new_score > old_score:
                self.pareto_front_valset[task_idx] = new_score
                self.program_at_pareto_front_valset[task_idx] = {new_program_idx}
                os.makedirs(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}"), exist_ok=True)
                with open(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}", f"iter_{self.i+1}_prog_{new_program_idx}.json"), 'w') as f:
                    json.dump(valset_outputs[task_idx], f, indent=4, default=json_default)
            elif new_score == old_score:
                self.program_at_pareto_front_valset[task_idx].add(new_program_idx)
        
        assert len(valset_subscores) == len(self.program_at_pareto_front_valset)

        if track_scores_on == 'train_val':
            train_val_weighted_agg_scores_for_all_programs = calculate_aggregate_trainval_scores(
                prog_ids=range(len(self.program_candidates)),
                train_scores=self.program_full_scores,
                val_scores=self.program_full_scores_val_set,
                trainset_len=len(trainset_subscores),
                valset_len=len(valset_subscores)
            )
            self.per_program_tracked_scores = train_val_weighted_agg_scores_for_all_programs
        else:
            self.per_program_tracked_scores = self.program_full_scores_val_set

        linear_pareto_front_program_idx = idxmax(self.per_program_tracked_scores)

        return new_program_idx, linear_pareto_front_program_idx

def calculate_aggregate_trainval_scores(prog_ids, train_scores, val_scores, trainset_len, valset_len):
    train_val_weighted_agg_scores_for_all_programs = []
    for prog_idx in prog_ids:
        train_score = train_scores[prog_idx]
        val_score = val_scores[prog_idx]
        train_weight = trainset_len / (trainset_len + valset_len)
        val_weight = valset_len / (trainset_len + valset_len)
        weighted_agg_score = train_weight * train_score + val_weight * val_score
        train_val_weighted_agg_scores_for_all_programs.append(weighted_agg_score)
    
    return train_val_weighted_agg_scores_for_all_programs

def json_default(x):
    """Default JSON encoder for objects that are not serializable by default."""
    try:
        return {**x}
    except:
        return repr(x)

def idxmax(lst):
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)

def update_pareto_front(new_prog_all_scores, new_program_idx, pareto_front, program_at_pareto_front):
    pareto_front = deepcopy(pareto_front)
    program_at_pareto_front = deepcopy(program_at_pareto_front)

    for score_idx, score in enumerate(new_prog_all_scores):
        if score > pareto_front[score_idx]:
            program_at_pareto_front[score_idx] = {new_program_idx}
            pareto_front[score_idx] = score
        elif score == pareto_front[score_idx]:
            program_at_pareto_front[score_idx].add(new_program_idx)
        else:
            pass

    return pareto_front, program_at_pareto_front

def capture_module_trace_with_feedback(
    module: dspy.Module,
    full_program: dspy.Module,
    evalset: list[dspy.Example],
    metric_fn: callable,
    logger,
    gepa_state: GEPAState,  # Can be None for GEPA_SGD
    skip_perfect_score: bool,
    perfect_score: float,
    failure_score: float,
    format_failure_score: float,
    feedback_func: callable,
    add_format_failure_as_feedback: bool,
    num_threads: int,
    rng: random.Random = None,  # Optional RNG for when gepa_state is None
):
    """
    Returns dataset_with_feedback, subsample_score, subsample_scores
    dataset_with_feedback is a list of dictionaries with keys: ['inputs', 'generated_output', 'feedback']
    """
    round_data = bootstrap_trace_data(
        program=full_program,
        dataset=evalset,
        metric=metric_fn,
        num_threads=num_threads,
        raise_on_error=False,
        capture_failed_parses=True,
        failure_score=failure_score,
        format_failure_score=format_failure_score,
    )

    # round_data is a list of dictionaries with keys: ['example', 'prediction', 'trace', 'example_ind', 'score']
    subscores = []
    # Use provided rng or gepa_state.rng1
    _rng = rng if rng is not None else (gepa_state.rng1 if gepa_state is not None else random.Random())

    ret = []
    for data in round_data:
        d = {}

        # Trace is [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        trace_instances_for_current_pred = [t for t in data["trace"] if hash(t[0].signature) == hash(module.signature)]

        if not add_format_failure_as_feedback:
            # If we are not adding format failure as feedback, we will only consider successful predictions
            trace_instances_for_current_pred = [t for t in trace_instances_for_current_pred if not isinstance(t[2], FailedPrediction)]
        
        if len(trace_instances_for_current_pred) == 0:
            iter_info = f"Iteration {gepa_state.i+1}: " if gepa_state is not None else ""
            logger.log(f"{iter_info}No trace instances found for module {module.signature}. Skipping.")
            continue

        selected_trace_instance = None
        for trace_instance in trace_instances_for_current_pred:
            if isinstance(trace_instance[2], FailedPrediction):
                selected_trace_instance = trace_instance

        if selected_trace_instance is None:
            # This means that all trace instances for current predictor are successful predictions
            if isinstance(data['prediction'], FailedPrediction):
                # This is coming from a different predictor, hence we don't have a good feedback for the current predictor
                continue
            selected_trace_instance = _rng.choice(trace_instances_for_current_pred)

        d['inputs'] = selected_trace_instance[1]
        d['generated_output'] = selected_trace_instance[2]

        if isinstance(selected_trace_instance[2], FailedPrediction):
            adapter = dspy.ChatAdapter()
            structure_instruction = ""
            for dd in adapter.format(module.signature, [], {}):
                structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
            feedback_text = f"Your output text failed to parse. Please ensure that your output follows the structure:\n{structure_instruction}"
            score = failure_score
            d['feedback'] = feedback_text
            d['score'] = score
            subscores.append(score)
        else:
            feedback_d = feedback_func(
                predictor_output=d['generated_output'], 
                predictor_inputs=d['inputs'], 
                module_inputs=data['example'],
                module_outputs=data['prediction'],
                captured_trace=data['trace'],
            )

            score, feedback_text = feedback_d["feedback_score"], feedback_d["feedback_text"]
            d['feedback'] = feedback_text
            d['score'] = data['score']  # 保存 overall score 到 trace
            subscores.append(data['score'])

        ret.append(d)

    iter_info = f"Iteration {gepa_state.i+1}: " if gepa_state is not None else ""

    if len(ret) == 0:
        logger.log(f"{iter_info}No valid predictions found for module {module.signature}. Skipping.")
        return None, None, None

    if skip_perfect_score and all(score >= perfect_score for score in subscores):
        logger.log(f"{iter_info}All scores are perfect. Skipping module {module.signature}.")
        return None, None, None
    
    return ret, sum(subscores), subscores

def write_eval_output_to_directory(eval_out, output_dir):
    for task_idx, score in enumerate(eval_out[2]):
        os.makedirs(os.path.join(output_dir, f"task_{task_idx}"), exist_ok=True)
        with open(os.path.join(output_dir, f"task_{task_idx}", f"iter_{0}_prog_0.json"), 'w') as f:
            json.dump(eval_out[1][task_idx], f, indent=4, default=json_default)

def initialize_wandb(wandb_api_key: str = None, run_dir: str = None):
    try:
        import wandb
        if wandb_api_key:
            wandb.login(key=wandb_api_key, verify=True)
        else:
            wandb.login()
    except ImportError:
        raise ImportError("wandb is not installed. Please install it or set use_wandb=False.")
    except Exception as e:
        raise RuntimeError(f"Error logging into wandb: {e}")
    
    wandb_run = wandb.init(
        project="gepa",
        dir=os.path.join(run_dir, "wandb"),
        name=run_dir,
    )
    return wandb_run

def initialize_gepa_state(gepa_state_to_use, run_dir, logger, base_dspy_program, trainset_evaluator, valset_evaluator, seed, run_linearized_gepa, track_scores_on, train_val_size):
    if gepa_state_to_use is None:
        if os.path.exists(os.path.join(run_dir, "gepa_state.bin")) and os.path.exists(os.path.join(run_dir, "prog_candidates")):
            logger.log("Loading gepa state from run dir")
            gepa_state = GEPAState.load(run_dir)
        else:
            num_evals_run = 0
            if track_scores_on == 'train_val':
                try:
                    eval_out = trainset_evaluator(base_dspy_program)
                    num_evals_run += len(eval_out[2])
                except Exception as e:
                    logger.log(f"Exception during eval: {e}")
                    logger.log(traceback.format_exc())
                    raise e
                write_eval_output_to_directory(eval_out, os.path.join(run_dir, "generated_best_outputs"))
            else:
                eval_out = (None, [], [])

            valset_out = valset_evaluator(base_dspy_program)
            write_eval_output_to_directory(valset_out, os.path.join(run_dir, "generated_best_outputs_valset"))
            num_evals_run += len(valset_out[2])

            gepa_state = GEPAState(
                base_dspy_program, 
                eval_out,
                valset_out,
                seed,
                track_scores_on=track_scores_on,
                run_linearized_gepa=run_linearized_gepa,
            )

            gepa_state.num_full_ds_evals = 1
            gepa_state.total_num_evals_per_trainval_instance = num_evals_run / train_val_size
            gepa_state.total_num_evals = num_evals_run

    else:
        gepa_state = gepa_state_to_use

    return gepa_state

def find_dominator_programs(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    uniq_progs = []
    for front in new_program_at_pareto_front_valset:
        uniq_progs.extend(front)
    uniq_progs = set(uniq_progs)
    return list(uniq_progs)

def select_program_candidate_from_pareto_front(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs, rng):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    program_frequency_in_validation_pareto_front = {}
    for testcase_pareto_front in new_program_at_pareto_front_valset:
        for prog_idx in testcase_pareto_front:
            if prog_idx not in program_frequency_in_validation_pareto_front:
                program_frequency_in_validation_pareto_front[prog_idx] = 0
            program_frequency_in_validation_pareto_front[prog_idx] += 1
    
    sampling_list = [prog_idx for prog_idx, freq in program_frequency_in_validation_pareto_front.items() for _ in range(freq)]
    assert len(sampling_list) > 0
    curr_prog_id = rng.choice(sampling_list)
    return curr_prog_id

def log_detailed_metrics_after_discovering_new_program(logger, gepa_state, valset_score, new_prog_all_scores, full_score, new_program_idx, valset_subscores, new_instruction, use_wandb, linear_pareto_front_program_idx, track_scores_on):
    best_prog_as_per_agg_score = idxmax(gepa_state.per_program_tracked_scores)
    best_prog_as_per_agg_score_valset = idxmax(gepa_state.program_full_scores_val_set)
    if gepa_state.program_full_scores:
        best_prog_as_per_agg_score_trainset = idxmax(gepa_state.program_full_scores)
    else:
        best_prog_as_per_agg_score_trainset = None

    logger.log(f"Iteration {gepa_state.i+1}: Full valset score for new program: {valset_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Full trainset score for new program: {full_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Full train_val score for new program: {gepa_state.per_program_tracked_scores[new_program_idx]}")
    logger.log(f"Iteration {gepa_state.i+1}: Individual valset scores for new program: {valset_subscores}")
    logger.log(f"Iteration {gepa_state.i+1}: Individual trainset scores for new program: {new_prog_all_scores}")
    logger.log(f"Iteration {gepa_state.i+1}: New valset pareto front scores: {gepa_state.pareto_front_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: New trainset pareto front scores: {gepa_state.pareto_front}")
    logger.log(f"Iteration {gepa_state.i+1}: Full valset pareto front score: {sum(gepa_state.pareto_front_valset)/len(gepa_state.pareto_front_valset)}")
    logger.log(f"Iteration {gepa_state.i+1}: Updated valset pareto front programs: {gepa_state.program_at_pareto_front_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: Best valset aggregate score so far: {max(gepa_state.program_full_scores_val_set)}")
    logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on train_val: {best_prog_as_per_agg_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on valset: {best_prog_as_per_agg_score_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: Best score on valset: {gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset]}")
    logger.log(f"Iteration {gepa_state.i+1}: Best score on train_val: {gepa_state.per_program_tracked_scores[best_prog_as_per_agg_score]}")
    logger.log(f"Iteration {gepa_state.i+1}: Linear pareto front program index: {linear_pareto_front_program_idx}")
    logger.log(f"Iteration {gepa_state.i+1}: New program candidate index: {new_program_idx}")

    if track_scores_on == 'train_val':
        logger.log(f"Iteration {gepa_state.i+1}: Full trainset pareto front score: {sum(gepa_state.pareto_front)/len(gepa_state.pareto_front)}")
        logger.log(f"Iteration {gepa_state.i+1}: Updated trainset pareto front programs: {gepa_state.program_at_pareto_front}")
        logger.log(f"Iteration {gepa_state.i+1}: Best trainset aggregate score so far: {max(gepa_state.program_full_scores)}")
        logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on trainset: {best_prog_as_per_agg_score_trainset}")
        logger.log(f"Iteration {gepa_state.i+1}: Best score on trainset: {gepa_state.program_full_scores[best_prog_as_per_agg_score_trainset]}")

    wandb_logs = {
        "iteration": gepa_state.i+1,
        "full_valset_score": valset_score,
        "full_train_val_score": gepa_state.per_program_tracked_scores[new_program_idx],
        "new_instruction": new_instruction,
        "new_program_idx": new_program_idx,
        "valset_pareto_front_scores": gepa_state.pareto_front_valset,
        "individual_valset_score_new_program": valset_subscores,
        "valset_pareto_front_agg": sum(gepa_state.pareto_front_valset)/len(gepa_state.pareto_front_valset),
        "valset_pareto_front_programs": gepa_state.program_at_pareto_front_valset,
        "best_valset_agg_score": max(gepa_state.program_full_scores_val_set),
        "linear_pareto_front_program_idx": linear_pareto_front_program_idx,
        "best_program_as_per_agg_score": best_prog_as_per_agg_score,
        "best_program_as_per_agg_score_valset": best_prog_as_per_agg_score_valset,
        "best_score_on_valset": gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset],
        "best_score_on_train_val": gepa_state.per_program_tracked_scores[best_prog_as_per_agg_score],
    }

    if track_scores_on == 'train_val':
        wandb_logs.update({
            "full_trainset_score": full_score,
            "trainset_pareto_front_scores": gepa_state.pareto_front,
            "individual_trainset_score_new_program": new_prog_all_scores,
            "trainset_pareto_front_agg": sum(gepa_state.pareto_front)/len(gepa_state.pareto_front),
            "trainset_pareto_front_programs": gepa_state.program_at_pareto_front,
            "best_trainset_agg_score": max(gepa_state.program_full_scores),
            "best_program_as_per_agg_score_trainset": best_prog_as_per_agg_score_trainset,
            "best_score_on_trainset": gepa_state.program_full_scores[best_prog_as_per_agg_score_trainset],
        })

    if use_wandb:
        wandb.log(wandb_logs, step=gepa_state.i+1)
