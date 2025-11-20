"""
LiveBench Math with Trace Optimization
Adapted from run_hotpotqa_trace.py for LiveBench Math.

Key Features:
- Uses LiveBenchMath dataset
- Optimizes prompt templates for math problem solving
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
from gepa_artifact.benchmarks.livebench_math.livebenchmath_data import LiveBenchMathBench
from gepa_artifact.benchmarks.livebench_math import metric, metric_with_feedback

@model
class LiveBenchMathTrace(LLMCallable):
    """Traced version of LiveBench Math"""
    
    def __init__(self):
        super().__init__()
        
        self.generate_response_template = ParameterNode(
            "Given the fields `question`, produce the fields `answer`.",
            trainable=True,
            description="Template for solving math problems."
        )
    
    def forward(self, question):
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"question": "str"},
            output_fields={"reasoning": "str", "answer": "str"},
            instruction=self.generate_response_template,
            values={"question": question}
        )
        
        response = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        answer = parse_output_field(response, "answer")
        
        return {
            "answer": answer,
            "response": response
        }

class LiveBenchMathBaseline(LLMCallableBaseline):
    """Baseline version of LiveBench Math"""
    
    def __init__(self):
        super().__init__()
        self.generate_response_template = "Given the fields `question`, produce the fields `answer`."
    
    def forward(self, question):
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"question": "str"},
            output_fields={"reasoning": "str", "answer": "str"},
            instruction=self.generate_response_template,
            values={"question": question}
        )
        
        response = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        answer = parse_output_field_baseline(response, "answer")
        
        return {
            "answer": answer,
            "response": response
        }

def eval_metric_wrapper(example, prediction):
    if hasattr(prediction, 'data'):
        prediction = prediction.data
        
    if isinstance(prediction, dict):
        pred_answer = prediction.get('answer', '')
    else:
        pred_answer = prediction
        
    dspy_pred = dspy.Prediction(answer=pred_answer)
    return metric(example, dspy_pred)

def generate_feedback(example, prediction, result):
    if isinstance(result, dict):
        pred_answer = result.get('answer', '')
    else:
        pred_answer = result

    if hasattr(pred_answer, 'data'):
        pred_answer = pred_answer.data
        
    dspy_pred = dspy.Prediction(answer=pred_answer)
    feedback_result = metric_with_feedback(example, dspy_pred)
    
    return feedback_result.feedback, feedback_result.score

def main(dataset_mode="lite", seed=None, num_steps=20, run_test_baseline=False, optimizer="OptoPrime", save_dir=None):
    print("Loading LiveBench Math dataset...")
    
    if seed is not None:
        random.seed(seed)
    
    # Use save_dir if provided, otherwise use current directory
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    
    benchmark = LiveBenchMathBench(dataset_mode=dataset_mode)
    
    def example_to_dict(ex):
        # Ensure we preserve all fields needed (question, question_d, etc)
        return dict(ex)
        
    trainset = [example_to_dict(ex) for ex in benchmark.train_set]
    valset = [example_to_dict(ex) for ex in benchmark.val_set]
    testset = [example_to_dict(ex) for ex in benchmark.test_set]
    
    print(f"Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    
    model_instance = LiveBenchMathTrace()
    
    if run_test_baseline:
        evaluate_baseline_concurrent(
            testset=testset,
            forward_fn=lambda x: LiveBenchMathBaseline().forward(x),
            eval_fn=eval_metric_wrapper,
            input_key="question"
        )
        return
        
    print(f"Running optimization with {optimizer}...")
    optimized_model = train_optimization_loop(
        model_instance, 
        trainset, 
        optimizer_config={"optimizer": "answer"},
        feedback_fn=generate_feedback,
        input_key="question",
        num_steps=num_steps,
        optimizer_type=optimizer,
        save_dir=save_dir
    )
        
    optimized_baseline = LiveBenchMathBaseline()
    optimized_baseline.generate_response_template = optimized_model.generate_response_template.data
    
    evaluate_baseline_concurrent(
        testset=testset,
        forward_fn=lambda x: optimized_baseline.forward(x),
        eval_fn=eval_metric_wrapper,
        input_key="question"
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
