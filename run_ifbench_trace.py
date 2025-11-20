"""
IFBench Instruction Following with Trace Optimization
Adapted from run_hotpotqa_trace.py for IFBench benchmark.

Key Features:
- Uses IFBench dataset
- Optimizes prompt templates for instruction following (2 stages)
- Uses instruction following metric with feedback
"""

import os
import sys
import re
from textwrap import dedent
from typing import List
import json
import random

# Add the project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gepa_artifact'))

# Trace imports
from opto.trace.nodes import node, GRAPH, ParameterNode, MessageNode, ExceptionNode
MessageNode.__format__ = lambda self, format_spec: str(self.data)
from opto.trace.bundle import bundle
from opto.trace.modules import model

# Common utilities
from trace_utils import (
    LLMCallable, 
    LLMCallableBaseline, 
    create_dspy_prompt, 
    create_dspy_prompt_baseline, 
    parse_output_field, 
    parse_output_field_baseline,
    train_optimization_loop,
    evaluate_baseline_concurrent
)

# Benchmark specific imports
import dspy
from gepa_artifact.benchmarks.IFBench.ifbench_data import IFBench
from gepa_artifact.benchmarks.IFBench.ifbench_metric import metric, metric_with_feedback

@model
class IFBenchTrace(LLMCallable):
    """Traced version of IFBench using Trace library"""
    
    def __init__(self):
        super().__init__()
        
        # Stage 1: Generate initial response
        self.generate_response_template = ParameterNode(
            "Given the fields `query`, produce the fields `response`.",
            trainable=True,
            description="Template for generating initial response to the instruction."
        )
        
        # Stage 2: Ensure correctness / refinement
        self.ensure_correct_template = ParameterNode(
            "Given the fields `query`, `response`, produce the fields `final_response`.",
            trainable=True,
            description="Template for refining the response to ensure all constraints are met."
        )
    
    def forward(self, query):
        # Stage 1
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"query": "str"},
            output_fields={"reasoning": "str", "response": "str"},
            instruction=self.generate_response_template,
            values={"query": query}
        )
        
        response_1_raw = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        response_1 = parse_output_field(response_1_raw, "response")
        
        # Stage 2
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"query": "str", "response": "str"},
            output_fields={"reasoning": "str", "final_response": "str"},
            instruction=self.ensure_correct_template,
            values={"query": query, "response": response_1}
        )
        
        response_2_raw = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        final_response = parse_output_field(response_2_raw, "final_response")
        
        return {
            "response": response_1,
            "final_response": final_response
        }

class IFBenchBaseline(LLMCallableBaseline):
    """Baseline version of IFBench WITHOUT Trace library"""
    
    def __init__(self):
        super().__init__()
        self.generate_response_template = "Given the fields `query`, produce the fields `response`."
        self.ensure_correct_template = "Given the fields `query`, `response`, produce the fields `final_response`."
    
    def forward(self, query):
        # Stage 1
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"query": "str"},
            output_fields={"reasoning": "str", "response": "str"},
            instruction=self.generate_response_template,
            values={"query": query}
        )
        
        response_1_raw = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        response_1 = parse_output_field_baseline(response_1_raw, "response")
        
        # Stage 2
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"query": "str", "response": "str"},
            output_fields={"reasoning": "str", "final_response": "str"},
            instruction=self.ensure_correct_template,
            values={"query": query, "response": response_1}
        )
        
        response_2_raw = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        final_response = parse_output_field_baseline(response_2_raw, "final_response")
        
        return {
            "response": final_response # Return final response as response for metric
        }

def eval_metric_wrapper(example, prediction):
    if hasattr(prediction, 'data'):
        prediction = prediction.data
        
    # metric expects example (dict) and prediction (dspy.Prediction)
    # prediction should have 'response' field
    
    # If prediction is dict from forward
    if isinstance(prediction, dict):
        pred_response = prediction.get('final_response', prediction.get('response', ''))
    else:
        pred_response = prediction
        
    dspy_pred = dspy.Prediction(response=pred_response)
    
    # Convert example dict to dspy.Example so metric can access attributes
    dspy_example = dspy.Example(**example).with_inputs('prompt', 'instruction_id_list', 'kwargs')
    
    return metric(dspy_example, dspy_pred)

def generate_feedback(example, prediction, result):
    # Use metric_with_feedback to get detailed feedback
    if isinstance(result, dict):
        pred_response = result.get('final_response', result.get('response', ''))
    else:
        pred_response = result

    if hasattr(pred_response, 'data'):
        pred_response = pred_response.data
        
    dspy_pred = dspy.Prediction(response=pred_response)
    
    # Convert example dict to dspy.Example so metric can access attributes
    dspy_example = dspy.Example(**example).with_inputs('prompt', 'instruction_id_list', 'kwargs')
    
    feedback_result = metric_with_feedback(dspy_example, dspy_pred)
    score = feedback_result.score
    feedback_text = feedback_result.feedback
    
    return feedback_text, score

def main(dataset_mode="lite", seed=None, num_steps=20, run_test_baseline=False, optimizer="OptoPrime", save_dir=None):
    print("Loading IFBench dataset...")
    
    if seed is not None:
        random.seed(seed)
    
    # Use save_dir if provided, otherwise use current directory
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    
    benchmark = IFBench(dataset_mode=dataset_mode)
    
    # IFBench examples are dicts with 'prompt' and other fields
    # We need to ensure consistency
    def example_to_dict(ex):
        # Assuming ex is dspy.Example or similar
        return dict(ex)
        
    trainset = [example_to_dict(ex) for ex in benchmark.train_set]
    valset = [example_to_dict(ex) for ex in benchmark.val_set]
    testset = [example_to_dict(ex) for ex in benchmark.test_set]
    
    print(f"Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    
    model_instance = IFBenchTrace()
    
    if run_test_baseline:
        evaluate_baseline_concurrent(
            testset=testset,
            forward_fn=lambda x: IFBenchBaseline().forward(x),
            eval_fn=eval_metric_wrapper,
            input_key="prompt"
        )
        return
        
    print(f"Running optimization with {optimizer}...")
    optimized_model = train_optimization_loop(
        model_instance, 
        trainset, 
        optimizer_config={
            'stage1': 'response',
            'stage2': 'final_response'
        },
        feedback_fn=generate_feedback,
        input_key="prompt",
        num_steps=num_steps,
        optimizer_type=optimizer,
        save_dir=save_dir
    )
        
    optimized_baseline = IFBenchBaseline()
    optimized_baseline.generate_response_template = optimized_model.generate_response_template.data
    optimized_baseline.ensure_correct_template = optimized_model.ensure_correct_template.data
    
    evaluate_baseline_concurrent(
        testset=testset,
        forward_fn=lambda x: optimized_baseline.forward(x),
        eval_fn=eval_metric_wrapper,
        input_key="prompt"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-mode", type=str, default="lite")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--run-test-baseline", action="store_true")
    parser.add_argument("--optimizer", type=str, default="OptoPrime", choices=["OptoPrime", "TextGrad"],
                        help="Optimizer to use for training (default: OptoPrime)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save optimized parameters (default: current directory)")
    args = parser.parse_args()
    
    main(args.dataset_mode, args.seed, args.num_steps, args.run_test_baseline, args.optimizer, args.save_dir)
