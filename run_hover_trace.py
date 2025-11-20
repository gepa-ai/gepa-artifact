"""
Hover Multi-Hop Retrieval with Trace Optimization
Adapted from run_hotpotqa_trace.py for Hover benchmark.

Key Features:
- Uses HoverBench dataset
- Optimizes prompt templates for summaries and queries
- Uses discrete retrieval evaluation
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
from gepa_artifact.benchmarks.hover.hover_data import hoverBench
from gepa_artifact.benchmarks.hover.hover_program import search
from gepa_artifact.benchmarks.hover.hover_utils import discrete_retrieval_eval

@model
class HoverTrace(LLMCallable):
    """Traced version of Hover using Trace library"""
    
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        
        # Define trainable prompt templates
        self.summarize1_template = ParameterNode(
            "Given the fields `claim`, `passages`, produce the fields `summary`.",
            trainable=True,
            description="Template for summarizing first hop passages relevant to the claim."
        )
        
        self.query_hop2_template = ParameterNode(
            "Given the fields `claim`, `summary_1`, produce the fields `query`.",
            trainable=True,
            description="Template for generating second hop query."
        )
        
        self.summarize2_template = ParameterNode(
            "Given the fields `claim`, `context`, `passages`, produce the fields `summary`.",
            trainable=True,
            description="Template for summarizing second hop passages with context."
        )
        
        self.query_hop3_template = ParameterNode(
            "Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.",
            trainable=True,
            description="Template for generating third hop query."
        )
    
    def retrieve_k(self, query, k=None):
        if k is None:
            k = self.k
        if hasattr(query, 'data'):
            query = query.data
        return search(query, k=k)
    
    def format_passages(self, passages):
        return "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    
    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"claim": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize1_template,
            values={"claim": claim, "passages": self.format_passages(hop1_docs)}
        )
        response_1 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_1 = parse_output_field(response_1, "summary")
        
        # HOP 2
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"claim": "str", "summary_1": "str"},
            output_fields={"reasoning": "str", "query": "str"},
            instruction=self.query_hop2_template,
            values={"claim": claim, "summary_1": summary_1}
        )
        response_2 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        hop2_query = parse_output_field(response_2, "query")
        
        hop2_docs = self.retrieve_k(hop2_query).passages
        
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"claim": "str", "context": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize2_template,
            values={"claim": claim, "context": summary_1, "passages": self.format_passages(hop2_docs)}
        )
        response_3 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_2 = parse_output_field(response_3, "summary")
        
        # HOP 3
        system_prompt, user_prompt = create_dspy_prompt(
            input_fields={"claim": "str", "summary_1": "str", "summary_2": "str"},
            output_fields={"reasoning": "str", "query": "str"},
            instruction=self.query_hop3_template,
            values={"claim": claim, "summary_1": summary_1, "summary_2": summary_2}
        )
        response_4 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        hop3_query = parse_output_field(response_4, "query")
        
        hop3_docs = self.retrieve_k(hop3_query, k=10).passages
        
        return {
            "retrieved_docs": hop1_docs + hop2_docs + hop3_docs,
            "summary_1": summary_1,
            "hop2_query": hop2_query,
            "summary_2": summary_2,
            "hop3_query": hop3_query
        }

class HoverBaseline(LLMCallableBaseline):
    """Baseline version of Hover WITHOUT Trace library"""
    
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        self.summarize1_template = "Given the fields `claim`, `passages`, produce the fields `summary`."
        self.query_hop2_template = "Given the fields `claim`, `summary_1`, produce the fields `query`."
        self.summarize2_template = "Given the fields `claim`, `context`, `passages`, produce the fields `summary`."
        self.query_hop3_template = "Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`."
    
    def retrieve_k(self, query, k=None):
        if k is None:
            k = self.k
        return search(query, k=k)
    
    def format_passages(self, passages):
        return "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    
    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"claim": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize1_template,
            values={"claim": claim, "passages": self.format_passages(hop1_docs)}
        )
        response_1 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_1 = parse_output_field_baseline(response_1, "summary")
        
        # HOP 2
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"claim": "str", "summary_1": "str"},
            output_fields={"reasoning": "str", "query": "str"},
            instruction=self.query_hop2_template,
            values={"claim": claim, "summary_1": summary_1}
        )
        response_2 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        hop2_query = parse_output_field_baseline(response_2, "query")
        
        hop2_docs = self.retrieve_k(hop2_query).passages
        
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"claim": "str", "context": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize2_template,
            values={"claim": claim, "context": summary_1, "passages": self.format_passages(hop2_docs)}
        )
        response_3 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_2 = parse_output_field_baseline(response_3, "summary")
        
        # HOP 3
        system_prompt, user_prompt = create_dspy_prompt_baseline(
            input_fields={"claim": "str", "summary_1": "str", "summary_2": "str"},
            output_fields={"reasoning": "str", "query": "str"},
            instruction=self.query_hop3_template,
            values={"claim": claim, "summary_1": summary_1, "summary_2": summary_2}
        )
        response_4 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        hop3_query = parse_output_field_baseline(response_4, "query")
        
        hop3_docs = self.retrieve_k(hop3_query, k=10).passages
        
        return {
            "retrieved_docs": hop1_docs + hop2_docs + hop3_docs
        }

def eval_metric_wrapper(example, prediction):
    # discrete_retrieval_eval expects dspy.Prediction(retrieved_docs=...)
    # and example with 'supporting_facts'
    if hasattr(prediction, 'data'):
        prediction = prediction.data
    
    if isinstance(prediction, dict):
        retrieved_docs = prediction.get('retrieved_docs', [])
    else:
        retrieved_docs = prediction
        
    dspy_pred = dspy.Prediction(retrieved_docs=retrieved_docs)
    return discrete_retrieval_eval(example, dspy_pred)

def generate_feedback(example, prediction, result):
    # Here result is passed as full dict
    score = eval_metric_wrapper(example, result)
    
    correctness = (score >= 1.0)
    
    if correctness:
        feedback = "The retrieved documents contain all the supporting facts! Good job."
    else:
        # Identify missing docs
        gold_titles = set(
            map(
                dspy.evaluate.normalize_text,
                [doc["key"] for doc in example["supporting_facts"]],
            )
        )
        found_titles = set(
            map(
                dspy.evaluate.normalize_text,
                [c.split(" | ")[0] for c in result["retrieved_docs"]],
            )
        )
        missing = gold_titles - found_titles
        feedback = f"The retrieval is incomplete. You missed the following key documents: {', '.join(missing)}. Please improve the summaries and queries to retrieve these documents."
        
    return feedback, score

def main(dataset_mode="lite", seed=None, num_steps=20, run_test_baseline=False, optimizer="OptoPrime", save_dir=None):
    print("Loading Hover dataset...")
    
    if seed is not None:
        random.seed(seed)
    
    # Use save_dir if provided, otherwise use current directory
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    
    benchmark = hoverBench(dataset_mode=dataset_mode)
    
    def example_to_dict(ex):
        return {
            "claim": ex.claim,
            "supporting_facts": ex.supporting_facts,
            "label": ex.get("label", "")
        }
        
    trainset = [example_to_dict(ex) for ex in benchmark.train_set]
    valset = [example_to_dict(ex) for ex in benchmark.val_set]
    testset = [example_to_dict(ex) for ex in benchmark.test_set]
    
    print(f"Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    
    model_instance = HoverTrace(k=7)
    
    if run_test_baseline:
        evaluate_baseline_concurrent(
            testset=testset,
            forward_fn=lambda x: HoverBaseline(k=7).forward(x),
            eval_fn=eval_metric_wrapper,
            input_key="claim"
        )
        return
        
    print(f"Running optimization with {optimizer}...")
    optimized_model = train_optimization_loop(
        model_instance, 
        trainset, 
        optimizer_config={
            'summarize1': 'summary_1',
            'query_hop2': 'hop2_query',
            'summarize2': 'summary_2',
            'query_hop3': 'hop3_query'
        },
        feedback_fn=generate_feedback,
        input_key="claim",
        num_steps=num_steps,
        optimizer_type=optimizer,
        save_dir=save_dir
    )
        
    optimized_baseline = HoverBaseline(k=7)
    optimized_baseline.summarize1_template = optimized_model.summarize1_template.data
    optimized_baseline.query_hop2_template = optimized_model.query_hop2_template.data
    optimized_baseline.summarize2_template = optimized_model.summarize2_template.data
    optimized_baseline.query_hop3_template = optimized_model.query_hop3_template.data
    
    evaluate_baseline_concurrent(
        testset=testset, 
        forward_fn=lambda x: optimized_baseline.forward(x),
        eval_fn=eval_metric_wrapper,
        input_key="claim"
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
