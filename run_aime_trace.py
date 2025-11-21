"""
AIME Question Answering with Trace Optimization
Adapted from run_hotpotqa_trace.py for AIME benchmark.

Key Features:
- Uses AIMEBench dataset
- Optimizes prompt templates for CoT
- Supports train/val/test splits
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
from gepa_artifact.benchmarks.AIME.AIME_data import AIMEBench
from gepa_artifact.benchmarks.AIME import metric

@model
class AIMETrace(LLMCallable):
    """Traced version of AIME using Trace library"""
    
    def __init__(self):
        super().__init__()
        
        # Define trainable prompt template
        self.generate_response_template = ParameterNode(
            "Given the fields `problem`, produce the fields `answer`.",
            trainable=True,
            description="Template for solving AIME problems. Should encourage step-by-step reasoning."
        )
    
    def forward(self, problem):
        """Main forward pass for AIME"""
        
        # Create prompt using DSPy format
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"problem": "str"},
            output_fields={"reasoning": "str", "answer": "str"},
            instruction=self.generate_response_template,
            values={"problem": problem}
        )
        
        response = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        answer = parse_output_field(response, "answer")
        
        return {
            "answer": answer,
            "response": response
        }

class AIMEBaseline(LLMCallableBaseline):
    """Baseline version of AIME WITHOUT Trace library"""
    
    def __init__(self):
        super().__init__()
        self.generate_response_template = "Given the fields `problem`, produce the fields `answer`."
    
    def forward(self, problem):
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"problem": "str"},
            output_fields={"reasoning": "str", "answer": "str"},
            instruction=self.generate_response_template,
            values={"problem": problem}
        )
        
        response = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        answer = parse_output_field_baseline(response, "answer")
        
        return {
            "answer": answer,
            "response": response
        }

def eval_metric_wrapper(example, prediction):
    """Evaluate if prediction matches the expected answer using benchmark metric"""

    
    if isinstance(prediction, dict):
        prediction = prediction.get('answer', '')

    if hasattr(prediction, 'data'):
        prediction = prediction.data

    # The metric function expects a dspy.Prediction object with an 'answer' attribute
    # and an example dict-like object with 'answer'
    dspy_pred = dspy.Prediction(answer=prediction)
    
    # AIME metric requires integer conversion check
    return metric(example, dspy_pred)

def generate_feedback(example, prediction, result):
    """Generate feedback for the optimizer"""
    # Use result instead of prediction
    score = eval_metric_wrapper(example, result)
    
    correctness = (score == 1)
    
    if correctness:
        feedback = "The answer is correct! No need to change anything."
    else:
        feedback = f"The answer is wrong. We expect the output of your answer to be \"{example['answer']}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer."
        
        # Add solution if available (similar to metric_with_feedback in original)
        written_solution = example.get('solution', '')
        if written_solution:
            feedback += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return feedback, score

def main(dataset_mode="lite", seed=None, num_steps=20, run_test_baseline=False, use_params_file=None, optimizer="OptoPrime", save_dir=None):
    print("Loading AIME dataset...")
    
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to: {seed}")
    
    # Use save_dir if provided, otherwise use current directory
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    
    benchmark = AIMEBench(dataset_mode=dataset_mode)
    
    def example_to_dict(ex):
        return {
            "problem": ex.problem,
            "answer": ex.answer,
            "solution": ex.get("solution", "")
        }
    
    trainset = [example_to_dict(ex) for ex in benchmark.train_set]
    valset = [example_to_dict(ex) for ex in benchmark.val_set]
    testset = [example_to_dict(ex) for ex in benchmark.test_set]
    
    print(f"Loaded dataset splits:")
    print(f"  Train: {len(trainset)} examples")
    print(f"  Val: {len(valset)} examples")
    print(f"  Test: {len(testset)} examples")
    
    print("\nInitializing AIMETrace model...")
    model_instance = AIMETrace()
    
    if run_test_baseline:
        baseline_model = None
        if use_params_file:
            print(f"\nLoading optimized parameters from: {use_params_file}")
            with open(use_params_file, 'r') as f:
                optimized_params = json.load(f)
            
            # Create baseline model with optimized parameters
            baseline_model = AIMEBaseline()
            baseline_model.generate_response_template = optimized_params["generate_response_template"]
            print("✓ Loaded optimized parameters")
            print(f"  Template preview: {baseline_model.generate_response_template[:100]}...")
        else:
            baseline_model = AIMEBaseline()
        
        evaluate_baseline_concurrent(
            testset=testset, 
            forward_fn=lambda x: baseline_model.forward(x),
            eval_fn=eval_metric_wrapper,
            input_key="problem"
        )
        return
    
    print(f"\nRunning optimization with {optimizer}...")
    
    # Create untraced baseline model for validation
    untraced_model = AIMEBaseline()
    
    optimized_model = train_optimization_loop(
        model_instance,
        untraced_model_instance=untraced_model,
        trainset=trainset,
        optimizer_config={"optimizer": "answer"},
        feedback_fn=generate_feedback,
        eval_fn=eval_metric_wrapper,
        valset=valset,
        input_key="problem",
        num_steps=num_steps,
        optimizer_type=optimizer,
        save_dir=save_dir
    )
    
    print("\nCreating optimized baseline model for final evaluation (parallel)...")
    optimized_baseline = AIMEBaseline()
    optimized_baseline.generate_response_template = optimized_model.generate_response_template.data
    
    evaluate_baseline_concurrent(
        testset=testset, 
        forward_fn=lambda x: optimized_baseline.forward(x),
        eval_fn=eval_metric_wrapper,
        input_key="problem"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AIME with Trace Optimization")
    parser.add_argument("--dataset-mode", type=str, default="lite", choices=["lite", "full", "tiny", "test"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--run-test-baseline", action="store_true")
    parser.add_argument("--use-params-file", type=str, default=None,
                        help="Path to optimized parameters JSON file to use for testing")
    parser.add_argument("--optimizer", type=str, default="OptoPrime", choices=["OptoPrime", "TextGrad"],
                        help="Optimizer to use for training (default: OptoPrime)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save optimized parameters (default: current directory)")
    
    args = parser.parse_args()
    
    main(
        dataset_mode=args.dataset_mode, 
        seed=args.seed, 
        num_steps=args.num_steps,
        run_test_baseline=args.run_test_baseline,
        use_params_file=args.use_params_file,
        optimizer=args.optimizer,
        save_dir=args.save_dir
    )
