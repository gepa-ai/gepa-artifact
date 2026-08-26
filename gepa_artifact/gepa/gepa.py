
from ast import Set
import math
import os
import random
import traceback
from typing import Literal, Union
import dspy
import json

import dspy.teleprompt
import dspy.teleprompt.teleprompt
import wandb

from .instruction_proposal import ProposeNewInstructionModule
from dspy import Example
from typing import List, Optional, Set
from collections import Counter

from .gepa_utils import (
    GEPAState,
    idxmax,
    capture_module_trace_with_feedback,
    select_program_candidate_from_pareto_front,
    log_detailed_metrics_after_discovering_new_program,
    initialize_gepa_state,
    initialize_wandb,
    find_dominator_programs
)

from .merge_programs import (
    sample_and_attempt_merge_programs_by_common_predictors
)

class GEPA(dspy.teleprompt.teleprompt.Teleprompter):
    def __init__(
        self,
        named_predictor_to_feedback_fn_map: dict[str, callable],
        knowledgebase_qe,
        metric: callable,
        logger,
        run_dir: str,
        run_linearized_gepa: bool=True,
        skip_full_eval_in_linear_mode: bool=False,
        num_threads=None,
        num_iters=None,
        failure_score=0,
        perfect_score=1,
        teacher_lm: dspy.LM = None,
        use_wandb: bool = False,
        wandb_api_key: str = None,
        max_evals_per_trainval_instance=None,
        seed=0,
        skip_perfect_score=True,
        use_merge=False,
        max_merge_invocations=5,
        num_dspy_examples_per_gepa_step=3,
        max_metric_calls=None,
        set_for_merge_minibatch='train',  # 'train', 'val', or 'both'
        track_scores_on: Literal['val', 'train_val'] = 'train_val',
        add_format_failure_as_feedback: bool=False,
        predictor_subset: Optional[List[str]] = None,
    ):
        # Exactly one of max_metric_calls, max_evals_per_trainval_instance or num_iters should be set
        assert (max_metric_calls is not None) + (max_evals_per_trainval_instance is not None) + (num_iters is not None) == 1, "Exactly one of max_metric_calls, max_evals_per_trainval_instance or num_iters should be set. You set max_metric_calls={}, max_evals_per_trainval_instance={}, num_iters={}".format(
            max_metric_calls, max_evals_per_trainval_instance, num_iters
        )   

        self.named_predictor_to_feedback_fn_map = named_predictor_to_feedback_fn_map
        self.knowledgebase_qe = knowledgebase_qe
        self.metric_fn = metric
        self.logger = logger
        self.run_dir = run_dir
        self.run_linearized_gepa = run_linearized_gepa
        self.skip_full_eval_in_linear_mode = skip_full_eval_in_linear_mode
        self.num_threads = num_threads
        
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.teacher_lm = teacher_lm
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key

        # Run constraints
        self.num_iters = num_iters
        self.max_evals_per_trainval_instance = max_evals_per_trainval_instance
        self.max_metric_calls = max_metric_calls

        self.seed = seed
        self.skip_perfect_score = skip_perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.set_for_merge_minibatch = set_for_merge_minibatch
        if self.set_for_merge_minibatch in ['train', 'both']:
            assert track_scores_on == 'train_val', "track_scores_on should be 'train_val' if set_for_merge_minibatch is 'train' or 'both'. You set track_scores_on={}".format(track_scores_on)

        assert track_scores_on in ['val', 'train_val'], "track_scores_on should be either 'val' or 'train_val'. You set track_scores_on={}".format(track_scores_on)
        self.track_scores_on = track_scores_on

        self.valset_provided = None
        self.train_val_size = None

        self.num_dspy_examples_per_gepa_step = num_dspy_examples_per_gepa_step

        self.add_format_failure_as_feedback = add_format_failure_as_feedback

        self.predictor_subset = None
        if predictor_subset:
            self.predictor_subset = set()
            for name in predictor_subset:
                self.predictor_subset.add(name)
                if not name.endswith(".predict"):
                    self.predictor_subset.add(f"{name}.predict")
        
        self.shuffled_trainset_ids = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.gepa_state = None

    def compile(
        self, student, trainset, valset,
    ):
        if valset is not None:
            self.valset_provided = True
        
        assert trainset is not None, "Trainset must be provided"

        gepa_state = self.gepa(
            base_dspy_program=student,
            trainset=trainset,
            valset=valset,
        )

        best_prog_idx = idxmax(gepa_state.per_program_tracked_scores)
        best_prog = gepa_state.program_candidates[best_prog_idx]
        return best_prog

    def update_shuffled_trainset(self, original_trainset):
        self.shuffled_trainset_ids = list(range(len(original_trainset)))
        self.gepa_state.rng1.shuffle(self.shuffled_trainset_ids)
        for id in self.shuffled_trainset_ids:
            self.id_freqs[id] += 1

        num_to_pad = self.num_dspy_examples_per_gepa_step - (len(original_trainset) % self.num_dspy_examples_per_gepa_step)
        if num_to_pad > 0:
            # Select ids based on least frequent ids
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_trainset_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def select_training_sample_and_update_shuffled_trainset(
        self,
        original_trainset: List[Example],
        train_step_idx: int,
    ) -> List[Example]:
        base_idx = train_step_idx * self.num_dspy_examples_per_gepa_step
        if self.epoch == -1:
            curr_epoch = 0
        else:
            curr_epoch = base_idx // len(self.shuffled_trainset_ids)
        if curr_epoch > self.epoch:
            print(f"Updating shuffled trainset for epoch {curr_epoch}...")
            self.epoch = curr_epoch
            self.update_shuffled_trainset(original_trainset)

        assert len(self.shuffled_trainset_ids) >= self.num_dspy_examples_per_gepa_step, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is less than num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_gepa_step}"
        assert len(self.shuffled_trainset_ids) % self.num_dspy_examples_per_gepa_step == 0, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is not divisible by num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_gepa_step}"

        base_idx = base_idx % len(self.shuffled_trainset_ids)
        end_idx = base_idx + self.num_dspy_examples_per_gepa_step
        assert end_idx <= len(self.shuffled_trainset_ids), f"End index {end_idx} is out of bounds for shuffled trainset length {len(self.shuffled_trainset_ids)}"
        selected_ids = self.shuffled_trainset_ids[base_idx:end_idx]
        return selected_ids
    
    def select_eval_subsample_for_merged_program(
        self,
        scores1,
        scores2,
        rng: random.Random,
        num_subsample_ids: int = 5,
    ) -> List[int]:
        all_indices = set(range(len(scores1)))
        # Partitioning
        partition1_ids = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s1 > s2]
        partition2_ids = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s2 > s1]
        partition3_ids = [i for i in all_indices if i not in partition1_ids and i not in partition2_ids]
        # Set up sample sizes
        n_each = math.ceil(num_subsample_ids / 3)
        n1 = min(len(partition1_ids), n_each)
        n2 = min(len(partition2_ids), n_each)
        n3 = min(len(partition3_ids), num_subsample_ids - (n1 + n2))
        # Sample
        # rng = gepa_state.rng1
        selected = []
        if n1: selected += rng.sample(partition1_ids, k=n1)
        if n2: selected += rng.sample(partition2_ids, k=n2)
        if n3: selected += rng.sample(partition3_ids, k=n3)
        # Pad if needed to desired length, without duplicates if possible
        remaining = num_subsample_ids - len(selected)
        unused = list(all_indices - set(selected))
        if remaining > 0:
            if len(unused) >= remaining:
                selected += rng.sample(unused, k=remaining)
            else:
                # All unique exhausted; use replacement
                selected += rng.choices(list(all_indices), k=remaining)
        return selected[:num_subsample_ids]
    
    def get_pareto_front_programs(self, gepa_state: GEPAState) -> List[Set[int]]:
        return (
            gepa_state.program_at_pareto_front_valset + \
            gepa_state.program_at_pareto_front + \
            [
                # {idxmax(train_val_weighted_agg_scores_for_all_programs)}, # Add best aggregate program
                # {idxmax(gepa_state.program_full_scores_val_set)}, # Add best on valset
                # {idxmax(gepa_state.program_full_scores)}, # Add best on trainset
            ] # TODO: Think about whether this should be added or not. Make this configurable.
        ) if self.track_scores_on == 'train_val' else gepa_state.program_at_pareto_front_valset
    
    def select_next_candidate_to_update(self, gepa_state: GEPAState):
        # TODO: Update this method to use pareto front from both train and val sets configurable
        assert len(gepa_state.per_program_tracked_scores) == len(gepa_state.program_candidates)

        if not gepa_state.running_linearized_gepa:
            curr_prog_id = select_program_candidate_from_pareto_front(
                self.get_pareto_front_programs(gepa_state), 
                gepa_state.per_program_tracked_scores,
                gepa_state.rng1,
            )
        else:
            curr_prog_id = idxmax(gepa_state.per_program_tracked_scores)

        return curr_prog_id
    
    def run_full_eval_add_new_program_to_gepa_tree(
        self,
        new_program: dspy.Module,
        gepa_state: GEPAState,
        trainset_evaluator: dspy.Evaluate,
        valset_evaluator: dspy.Evaluate,
        parent_program_idx: List[int]
    ):
        num_metric_calls_by_discovery_of_new_program = gepa_state.total_num_evals

        # Calculate metrics for new program and update gepa state
        if self.track_scores_on == 'train_val':
            trainset_score, trainset_outputs, trainset_subscores = trainset_evaluator(new_program)
        else:
            assert self.track_scores_on == 'val', "track_scores_on should be either 'val' or 'train_val'. You set track_scores_on={}".format(self.track_scores_on)
            trainset_score, trainset_outputs, trainset_subscores = None, None, None
        valset_score, valset_outputs, valset_subscores = valset_evaluator(new_program)

        # We have run one full eval of the new program on train set and val set
        gepa_state.num_full_ds_evals += 1
        if self.track_scores_on == 'train_val':
            gepa_state.total_num_evals_per_trainval_instance += 1
        else:
            assert self.track_scores_on == 'val', "track_scores_on should be either 'val' or 'train_val'. You set track_scores_on={}".format(self.track_scores_on)
            gepa_state.total_num_evals_per_trainval_instance += (len(valset_subscores) / self.train_val_size)
        gepa_state.total_num_evals += len(trainset_subscores) + len(valset_subscores) if self.track_scores_on == 'train_val' else len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
            parent_program_idx=parent_program_idx, # TODO: Handle this better. Mark both parents
            new_program=new_program,
            trainset_score=trainset_score,
            trainset_outputs=trainset_outputs,
            trainset_subscores=trainset_subscores,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            track_scores_on=self.track_scores_on,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery_of_new_program
        )

        gepa_state.full_program_trace[-1]['new_program_idx'] = new_program_idx

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {gepa_state.i+1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=gepa_state,
            valset_score=valset_score,
            new_prog_all_scores=trainset_subscores,
            full_score=trainset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            new_instruction="Merged program",
            use_wandb=self.use_wandb,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            track_scores_on= self.track_scores_on,
        )

        return new_program_idx, linear_pareto_front_program_idx

    def gepa(
        self,
        base_dspy_program: dspy.Module, 
        trainset: list[dspy.Example], 
        valset: list[dspy.Example]=None,
        gepa_state_to_use: Union[GEPAState, None]=None,
    ):
        if self.use_wandb:
            wandb_run = initialize_wandb(wandb_api_key=self.wandb_api_key, run_dir=self.run_dir)

        if self.num_threads is None:
            self.num_threads = os.cpu_count()

        trainset_evaluator = dspy.Evaluate(
            devset=trainset,
            metric=self.metric_fn,
            num_threads=self.num_threads,
            return_all_scores=True,
            return_outputs=True,
            failure_score=self.failure_score,
            provide_traceback=True,
            max_errors=len(trainset) * 100  # Allow for many errors in the training set
        )

        self.train_val_size = len(trainset)
        if valset is None:
            valset = trainset
        else:
            self.train_val_size += len(valset)

        valset_evaluator = dspy.Evaluate(
            devset=valset,
            metric=self.metric_fn,
            num_threads=self.num_threads,
            return_all_scores=True,
            return_outputs=True,
            failure_score=self.failure_score,
            provide_traceback=True,
            max_errors=len(valset) * 100  # Allow for many errors in the validation set
        )

        gepa_state = initialize_gepa_state(
            gepa_state_to_use=gepa_state_to_use,
            run_dir=self.run_dir,
            logger=self.logger,
            base_dspy_program=base_dspy_program,
            trainset_evaluator=trainset_evaluator,
            valset_evaluator=valset_evaluator,
            seed=self.seed,
            run_linearized_gepa=self.run_linearized_gepa,
            track_scores_on=self.track_scores_on,
            train_val_size= self.train_val_size,
        )

        self.gepa_state = gepa_state
        
        if self.track_scores_on == 'train_val':
            assert len(gepa_state.pareto_front) == len(trainset)
        
        assert len(gepa_state.pareto_front_valset) == len(valset), f"Pareto front valset length {len(gepa_state.pareto_front_valset)} does not match valset length {len(valset)}"

        if self.use_wandb:
            # assert gepa_state.i + 1 == 0
            wandb.log({
                "base_program_full_trainset_score": gepa_state.program_full_scores[0] if self.track_scores_on == 'train_val' else None,
                "base_program_full_valset_score": gepa_state.program_full_scores_val_set[0],
                "iteration": gepa_state.i+1,
            })
        if self.track_scores_on == 'train_val':
            self.logger.log(f"Iteration {gepa_state.i+1}: Base program full trainset score: {gepa_state.program_full_scores[0]}")
        self.logger.log(f"Iteration {gepa_state.i+1}: Base program full valset score: {gepa_state.program_full_scores_val_set[0]}")

        merges_due = 0
        total_merges_tested = 0

        last_iter_found_new_program = False

        merges_performed = ([], [])

        while (
            (self.num_iters is None or gepa_state.num_full_ds_evals < self.num_iters) and
            (self.max_evals_per_trainval_instance is None or gepa_state.total_num_evals_per_trainval_instance < self.max_evals_per_trainval_instance) and
            (self.max_metric_calls is None or gepa_state.total_num_evals < self.max_metric_calls)
        ):
            assert gepa_state.is_consistent(), "GEPA state is inconsistent, please check the implementation"
            try:
                gepa_state.save(self.run_dir)
                gepa_state.i += 1
                gepa_state.full_program_trace.append({"i": gepa_state.i})

                if merges_due > 0 and last_iter_found_new_program and self.use_merge:
                    last_iter_found_new_program = False
                    gepa_state.full_program_trace[-1]['invoked_merge'] = True

                    pareto_front_programs = self.get_pareto_front_programs(gepa_state)

                    merge_candidates = find_dominator_programs(pareto_front_programs, gepa_state.per_program_tracked_scores)
                    merge_output = sample_and_attempt_merge_programs_by_common_predictors(
                        agg_scores=gepa_state.per_program_tracked_scores,
                        rng=gepa_state.rng1,
                        merge_candidates=merge_candidates,
                        merges_performed=merges_performed,
                        program_candidates=gepa_state.program_candidates,
                        parent_program_for_candidate=gepa_state.parent_program_for_candidate,
                    )

                    if merge_output[0]:
                        gepa_state.full_program_trace[-1]['merged'] = True
                        success, new_program, id1, id2, ancestor = merge_output
                        assert success, "Merge output should be successful"
                        self.logger.log(f"Iteration {gepa_state.i+1}: Merged programs {id1} and {id2} via ancestor {ancestor}")
                        gepa_state.full_program_trace[-1]['merged_entities'] = (id1, id2, ancestor)

                        merges_performed[0].append((id1, id2, ancestor))

                        mini_devset = None

                        if self.set_for_merge_minibatch == 'train':
                            subsample_ids = self.select_eval_subsample_for_merged_program(
                                gepa_state.prog_candidate_train_subscores[id1],
                                gepa_state.prog_candidate_train_subscores[id2],
                                gepa_state.rng1,
                            )
                            mini_devset = [trainset[i] for i in subsample_ids]
                            id1_subsample_scores = [gepa_state.prog_candidate_train_subscores[id1][i] for i in subsample_ids]
                            id2_subsample_scores = [gepa_state.prog_candidate_train_subscores[id2][i] for i in subsample_ids]
                        elif self.set_for_merge_minibatch == 'val':
                            subsample_ids = self.select_eval_subsample_for_merged_program(
                                gepa_state.prog_candidate_val_subscores[id1],
                                gepa_state.prog_candidate_val_subscores[id2],
                                gepa_state.rng1,
                            )
                            mini_devset = [valset[i] for i in subsample_ids]
                            id1_subsample_scores = [gepa_state.prog_candidate_val_subscores[id1][i] for i in subsample_ids]
                            id2_subsample_scores = [gepa_state.prog_candidate_val_subscores[id2][i] for i in subsample_ids]
                        elif self.set_for_merge_minibatch == 'both':
                            subsample_ids = self.select_eval_subsample_for_merged_program(
                                gepa_state.prog_candidate_train_subscores[id1] + gepa_state.prog_candidate_val_subscores[id1],
                                gepa_state.prog_candidate_train_subscores[id2] + gepa_state.prog_candidate_val_subscores[id2],
                                gepa_state.rng1,
                            )
                            mini_devset = [trainset[i] if i < len(trainset) else valset[i - len(trainset)] for i in subsample_ids]
                            id1_subsample_scores = [gepa_state.prog_candidate_train_subscores[id1][i] if i < len(trainset) else gepa_state.prog_candidate_val_subscores[id1][i - len(trainset)] for i in subsample_ids]
                            id2_subsample_scores = [gepa_state.prog_candidate_train_subscores[id2][i] if i < len(trainset) else gepa_state.prog_candidate_val_subscores[id2][i - len(trainset)] for i in subsample_ids]
                        else:
                            self.logger.log(f"Iteration {gepa_state.i+1}: Unknown set for merge minibatch: {self.set_for_merge_minibatch}")
                            raise ValueError(f"Unknown set for merge minibatch: {self.set_for_merge_minibatch}. Should be 'train' or 'val'.")

                        gepa_state.full_program_trace[-1]['subsample_ids'] = subsample_ids

                        subsample_evaluator_args = {**trainset_evaluator.__dict__}
                        subsample_evaluator_args['devset'] = mini_devset
                        subsample_evaluator_args['return_outputs'] = True
                        subsample_evaluator_args['return_all_scores'] = True
                        subsample_evaluator_args['max_errors'] = len(subsample_ids) * 100
                        subsample_evaluator = dspy.Evaluate(**subsample_evaluator_args)

                        new_program_subsample_scores = subsample_evaluator(new_program)[2]

                        id1_subsample_score = sum(id1_subsample_scores)
                        id2_subsample_score = sum(id2_subsample_scores)
                        new_subsample_score = sum(new_program_subsample_scores)

                        gepa_state.full_program_trace[-1]['id1_subsample_scores'] = id1_subsample_scores
                        gepa_state.full_program_trace[-1]['id2_subsample_scores'] = id2_subsample_scores
                        gepa_state.full_program_trace[-1]['new_program_subsample_scores'] = new_program_subsample_scores

                        gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                        gepa_state.total_num_evals += len(subsample_ids)

                        if new_subsample_score >= max(id1_subsample_score, id2_subsample_score):
                            self.logger.log(f"Iteration {gepa_state.i+1}: New program subsample score {new_subsample_score} for merged program is better than min of both parents {id1_subsample_score} and {id2_subsample_score}. proceeding with full eval")
                        else:
                            self.logger.log(f"Iteration {gepa_state.i+1}: New program subsample score {new_subsample_score} is worse than both parent programs {id1_subsample_score} and {id2_subsample_score}, skipping merge")
                            continue

                        merges_due -= 1
                        total_merges_tested += 1

                        new_program_idx, linear_pareto_front_idx = self.run_full_eval_add_new_program_to_gepa_tree(
                            new_program=new_program,
                            gepa_state=gepa_state,
                            trainset_evaluator=trainset_evaluator,
                            valset_evaluator=valset_evaluator,
                            parent_program_idx=[id1, id2]
                        )
                        continue
                    else:
                        self.logger.log(f"Iteration {gepa_state.i+1}: No merge candidates found")
                
                last_iter_found_new_program = False

                curr_prog_id = self.select_next_candidate_to_update(gepa_state)
                curr_prog = gepa_state.program_candidates[curr_prog_id]

                gepa_state.full_program_trace[-1]['selected_program_candidate'] = curr_prog_id

                predictor_to_update_id = None
                predictor_name_to_update = None
                for _ in range(len(gepa_state.list_of_named_predictors)):
                    candidate_id = gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_id]
                    gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_id] = (
                        candidate_id + 1
                    ) % len(gepa_state.list_of_named_predictors)
                    candidate_name = gepa_state.list_of_named_predictors[candidate_id]
                    if self.predictor_subset is None or candidate_name in self.predictor_subset:
                        predictor_to_update_id = candidate_id
                        predictor_name_to_update = candidate_name
                        break

                if predictor_to_update_id is None:
                    self.logger.log(
                        f"Iteration {gepa_state.i+1}: Predictor subset did not match any program predictor, stopping"
                    )
                    break

                gepa_state.full_program_trace[-1]['predictor_to_update_id'] = predictor_to_update_id
                if predictor_name_to_update not in self.named_predictor_to_feedback_fn_map:
                    self.logger.log(f"Iteration {gepa_state.i+1}: Predictor {predictor_name_to_update} not in feedback map, skipping")
                    continue

                self.logger.log(f"Iteration {gepa_state.i+1}: Selected program candidate {curr_prog_id} with base score: {gepa_state.per_program_tracked_scores[curr_prog_id]}")
                self.logger.log(f"Iteration {gepa_state.i+1}: Updating predictor {predictor_name_to_update}")

                if self.use_wandb:
                    wandb.log({
                        "iteration": gepa_state.i+1,
                        "selected_program_candidate": curr_prog_id,
                        "predictor_to_update_id": predictor_to_update_id,
                    }, step=gepa_state.i+1)

                feedback_func = self.named_predictor_to_feedback_fn_map[predictor_name_to_update]
                module = None
                for m in curr_prog.named_predictors():
                    if m[0] == predictor_name_to_update:
                        module = m[1]
                        break
                assert module is not None

                subsample_ids = self.select_training_sample_and_update_shuffled_trainset(trainset, gepa_state.i)
                gepa_state.full_program_trace[-1]['subsample_ids'] = subsample_ids

                dataset_with_feedback, subsample_score, subsample_scores = capture_module_trace_with_feedback(
                    module, 
                    curr_prog, 
                    [trainset[i] for i in subsample_ids], 
                    self.metric_fn, 
                    self.logger, 
                    gepa_state,
                    self.skip_perfect_score,
                    self.perfect_score,
                    failure_score=self.failure_score,
                    format_failure_score=self.failure_score, # TODO: Get a proper value for this
                    feedback_func=feedback_func,
                    add_format_failure_as_feedback=self.add_format_failure_as_feedback,
                    num_threads=self.num_threads,
                )

                gepa_state.full_program_trace[-1]['subsample_scores'] = subsample_scores

                if dataset_with_feedback is None or subsample_score is None:
                    self.logger.log(f"Iteration {gepa_state.i+1}: No feedback samples, skipping")
                    continue

                if self.use_wandb:
                    wandb.log({
                        "subsample_score": subsample_score,
                    }, step=gepa_state.i+1)

                instruction_propose_module = ProposeNewInstructionModule(
                    base_program=module, 
                    instruction_lm=self.teacher_lm or dspy.dsp.utils.settings.lm or curr_prog.get_lm(),
                    dataset_with_feedback=dataset_with_feedback, 
                    knowledgebase_qe=self.knowledgebase_qe)
                if self.teacher_lm is not None:
                    instruction_propose_module.instruction_propose_module.set_lm(self.teacher_lm)
                try:
                    output = instruction_propose_module.compile()
                    with open(os.path.join(self.run_dir, "instruction_proposer_inpouts.jsonl"), 'a') as f:
                        f.write(json.dumps(output, default=lambda x: {**x}) + "\n")
                    new_instruction = output['new_instruction']
                    module_output = output['module_output']
                    kb_info = output['kb_info']
                except Exception as e:
                    self.logger.log(f"Iteration {gepa_state.i+1}: Exception during instruction proposal: {e}")
                    self.logger.log(traceback.format_exc())

                    continue
                self.logger.log(f"Iteration {gepa_state.i+1}: Info retrieved from knowledge base: {kb_info}")
                self.logger.log(f"Iteration {gepa_state.i+1}: Proposed new instruction: {new_instruction}")

                curr_prog_lm = curr_prog.get_lm()
                new_program = curr_prog.deepcopy()
                new_program.set_lm(curr_prog_lm)
                new_program.named_predictors()[predictor_to_update_id][1].signature = new_program.named_predictors()[predictor_to_update_id][1].signature.with_instructions(new_instruction)

                gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                gepa_state.total_num_evals += len(subsample_ids)

                # In pure SGD mode: skip validation, directly accept new prompt and move to next iteration
                if self.skip_full_eval_in_linear_mode and gepa_state.running_linearized_gepa:
                    self.logger.log(f"Iteration {gepa_state.i+1}: Pure SGD mode - directly accepting new prompt without validation")
                    # Directly replace the current program with the new one
                    gepa_state.program_candidates[curr_prog_id] = new_program
                    # Use subsample_score from trace generation as proxy
                    gepa_state.per_program_tracked_scores[curr_prog_id] = subsample_score
                    last_iter_found_new_program = True
                    continue

                # Original validation logic: re-run minibatch with new prompt to check if it's better
                subsample_evaluator_args = {**trainset_evaluator.__dict__}
                subsample_evaluator_args['devset'] = [trainset[i] for i in subsample_ids]
                subsample_evaluator_args['return_outputs'] = True
                subsample_evaluator_args['return_all_scores'] = True
                subsample_evaluator_args['max_errors'] = len(subsample_ids) * 100
                subsample_evaluator = dspy.Evaluate(**subsample_evaluator_args)
                new_subsample_scores = subsample_evaluator(new_program)[2]
                new_subsample_score = sum(new_subsample_scores)

                gepa_state.full_program_trace[-1]['new_subsample_scores'] = new_subsample_scores

                # TODO: How should this be incremented? What should be the denominator?
                gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                gepa_state.total_num_evals += len(subsample_ids)

                self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score: {new_subsample_score}")
                if self.use_wandb:
                    wandb.log({
                        "new_subsample_score": new_subsample_score,
                    }, step=gepa_state.i+1)
                
                if new_subsample_score <= subsample_score:
                    self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is not better, skipping")
                    continue

                last_iter_found_new_program = True

                self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is better, going from {subsample_score} to {new_subsample_score}, updating program candidate!")

                # Run full evaluation and add to GEPA tree (only reached when not in pure SGD mode)
                new_program_idx, linear_pareto_front_idx = self.run_full_eval_add_new_program_to_gepa_tree(
                    new_program=new_program,
                    gepa_state=gepa_state,
                    trainset_evaluator=trainset_evaluator,
                    valset_evaluator=valset_evaluator,
                    parent_program_idx=[curr_prog_id]
                )

                if self.use_merge and total_merges_tested < self.max_merge_invocations:
                    merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {gepa_state.i+1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                continue
        
        gepa_state.save(self.run_dir)

        return gepa_state
