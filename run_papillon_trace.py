"""
PAPILLON Privacy Benchmark with Trace Optimization
Adapted from run_hotpotqa_trace.py for Papillon benchmark.

Key Features:
- Uses Papillon dataset
- Optimizes prompt templates for privacy-preserving query transformation
- Simulates untrusted external model
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
from gepa_artifact.benchmarks.papillon.papillon_data import Papillon
from gepa_artifact.benchmarks.papillon.papillon_utils import compute_overall_score, compute_overall_score_with_feedback

@model
class PapillonTrace(LLMCallable):
    """Traced version of Papillon"""
    
    def __init__(self):
        super().__init__()
        
        # Main prompt templates
        self.craft_redacted_request_template = ParameterNode(
            "Given the fields `user_query`, produce the fields `llm_request`.",
            trainable=True,
            description="Template for creating a privacy-preserving request for an external LLM."
        )
        
        self.respond_to_query_template = ParameterNode(
            "Given the fields `related_llm_request`, `related_llm_response`, `user_query`, produce the fields `response`.",
            trainable=True,
            description="Template for responding to user query using external LLM response."
        )
        
        # Untrusted model simulator (not traced/optimized itself, but its usage is part of flow)
        # We use the same LLMCallable mechanism but with a fixed role
        self.untrusted_llm = LLMCallable(verbose=False) 
    
    def call_untrusted_model(self, request):
        # Simulates the external "untrusted" model
        # Using basic prompt
        return self.untrusted_llm.call_llm(
            system_prompt="You are a helpful assistant.",
            user_prompt=request
        )
    
    def forward(self, user_query):
        try:
            # Step 1: Craft redacted request
            system_prompt, user_prompt = create_dspy_prompt(
                input_fields={"user_query": "str"},
                output_fields={"reasoning": "str", "llm_request": "str"},
                instruction=self.craft_redacted_request_template,
                values={"user_query": user_query}
            )
            response_1 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            llm_request = parse_output_field(response_1, "llm_request")
            
            # Step 2: Call untrusted model
            llm_response = self.call_untrusted_model(llm_request)
            
            # Step 3: Respond to query
            # Note: respond_to_query is dspy.Predict in original, so NO reasoning field ideally
            # But requirement 1 says: "For dspy.Predict, just remove the reasoning field."
            system_prompt, user_prompt = create_dspy_prompt(
                input_fields={
                    "related_llm_request": "str", 
                    "related_llm_response": "str", 
                    "user_query": "str"
                },
                output_fields={"response": "str"},
                instruction=self.respond_to_query_template,
                values={
                    "related_llm_request": llm_request,
                    "related_llm_response": llm_response,
                    "user_query": user_query
                }
            )
            response_3 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            response = parse_output_field(response_3, "response")
            
            return {
                "llm_request": llm_request,
                "llm_response": llm_response,
                "response": response
            }
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return {
                "llm_request": "",
                "llm_response": "",
                "response": ""
            }

class PapillonBaseline(LLMCallableBaseline):
    """Baseline version of Papillon"""
    
    def __init__(self):
        super().__init__()
        self.craft_redacted_request_template = "Given the fields `user_query`, produce the fields `llm_request`."
        self.respond_to_query_template = "Given the fields `related_llm_request`, `related_llm_response`, `user_query`, produce the fields `response`."
        self.untrusted_llm = LLMCallableBaseline(verbose=False)
        
    def call_untrusted_model(self, request):
        return self.untrusted_llm.call_llm(
            system_prompt="You are a helpful assistant.",
            user_prompt=request
        )
        
    def forward(self, user_query):
        try:
            # Step 1
            system_prompt, user_prompt = create_dspy_prompt_baseline(
                input_fields={"user_query": "str"},
                output_fields={"reasoning": "str", "llm_request": "str"},
                instruction=self.craft_redacted_request_template,
                values={"user_query": user_query}
            )
            response_1 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            llm_request = parse_output_field_baseline(response_1, "llm_request")
            
            # Step 2
            llm_response = self.call_untrusted_model(llm_request)
            
            # Step 3 (No reasoning for Predict)
            system_prompt, user_prompt = create_dspy_prompt_baseline(
                input_fields={
                    "related_llm_request": "str", 
                    "related_llm_response": "str", 
                    "user_query": "str"
                },
                output_fields={"response": "str"},
                instruction=self.respond_to_query_template,
                values={
                    "related_llm_request": llm_request,
                    "related_llm_response": llm_response,
                    "user_query": user_query
                }
            )
            response_3 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            response = parse_output_field_baseline(response_3, "response")
            
            return {
                "llm_request": llm_request,
                "llm_response": llm_response,
                "response": response
            }
        except Exception:
            return {
                "llm_request": "",
                "llm_response": "",
                "response": ""
            }

def eval_metric_wrapper(example, prediction):
    if hasattr(prediction, 'data'):
        prediction = prediction.data
        
    # Prediction needs to have llm_request, llm_response, response
    # If prediction is dict (from forward)
    if isinstance(prediction, dict):
        dspy_pred = dspy.Prediction(
            llm_request=prediction.get('llm_request', ''),
            llm_response=prediction.get('llm_response', ''),
            response=prediction.get('response', '')
        )
    else:
        # Fallback
        dspy_pred = dspy.Prediction(llm_request="", llm_response="", response="")
    
    # Convert example dict to dspy.Example so metric can access attributes
    dspy_example = dspy.Example(**example).with_inputs('user_query', 'pii_str', 'target_response')
        
    return compute_overall_score(dspy_example, dspy_pred)

def generate_feedback(example, prediction, result):
    if isinstance(result, dict):
        dspy_pred = dspy.Prediction(
            llm_request=result.get('llm_request', ''),
            llm_response=result.get('llm_response', ''),
            response=result.get('response', '')
        )
    else:
        dspy_pred = dspy.Prediction(llm_request="", llm_response="", response="")
    
    # Convert example dict to dspy.Example so metric can access attributes
    dspy_example = dspy.Example(**example).with_inputs('user_query', 'pii_str', 'target_response')
        
    feedback_result = compute_overall_score_with_feedback(dspy_example, dspy_pred)
    return feedback_result.feedback, feedback_result.score

def main(dataset_mode="lite", seed=None, num_steps=20, run_test_baseline=False, optimizer="OptoPrime", save_dir=None):
    print("Loading Papillon dataset...")
    
    if seed is not None:
        random.seed(seed)
    
    # Use save_dir if provided, otherwise use current directory
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    
    benchmark = Papillon(dataset_mode=dataset_mode)
    
    def example_to_dict(ex):
        return dict(ex)
        
    trainset = [example_to_dict(ex) for ex in benchmark.train_set]
    valset = [example_to_dict(ex) for ex in benchmark.val_set]
    testset = [example_to_dict(ex) for ex in benchmark.test_set]
    
    print(f"Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    
    model_instance = PapillonTrace()
    
    if run_test_baseline:
        evaluate_baseline_concurrent(
            testset=testset,
            forward_fn=lambda x: PapillonBaseline().forward(x),
            eval_fn=eval_metric_wrapper,
            input_key="user_query"
        )
        return
        
    print(f"Running optimization with {optimizer}...")
    optimized_model = train_optimization_loop(
        model_instance, 
        trainset, 
        optimizer_config={
            # 'craft': 'llm_request',
            'respond': 'response'
        },
        feedback_fn=generate_feedback,
        input_key="user_query",
        num_steps=num_steps,
        optimizer_type=optimizer,
        save_dir=save_dir
    )
        
    optimized_baseline = PapillonBaseline()
    optimized_baseline.craft_redacted_request_template = optimized_model.craft_redacted_request_template.data
    optimized_baseline.respond_to_query_template = optimized_model.respond_to_query_template.data
    
    evaluate_baseline_concurrent(
        testset=testset,
        forward_fn=lambda x: optimized_baseline.forward(x),
        eval_fn=eval_metric_wrapper,
        input_key="user_query"
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
