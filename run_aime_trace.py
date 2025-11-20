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
from opto.optimizers import OptoPrime
from opto.trace.bundle import bundle
from opto.trace.modules import model
from opto.utils.llm import LLM

# Common utilities
from trace_utils import (
    LLMCallable, 
    LLMCallableBaseline, 
    create_dspy_prompt, 
    create_dspy_prompt_baseline, 
    parse_output_field, 
    parse_output_field_baseline
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
    if hasattr(prediction, 'data'):
        prediction = prediction.data
    
    # The metric function expects a dspy.Prediction object with an 'answer' attribute
    # and an example dict-like object with 'answer'
    dspy_pred = dspy.Prediction(answer=prediction)
    
    # AIME metric requires integer conversion check
    return metric(example, dspy_pred)

def generate_feedback(example, prediction, result):
    """Generate feedback for the optimizer"""
    score = eval_metric_wrapper(example, prediction)
    
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

def optimization_function(model_instance, trainset, valset=None, num_steps=20):
    """Main optimization function"""
    
    llm = LLM(model="gpt-4.1-mini")
    all_parameters = model_instance.parameters()
    
    # Single optimizer for the single output
    optimizer = OptoPrime(all_parameters, llm=llm)
    
    print(f"Starting optimization on {len(trainset)} training examples for {num_steps} steps...")
    
    for step in range(num_steps):
        print(f"\n{'='*80}")
        print(f"Optimization Step {step + 1}/{num_steps}")
        print(f"{'='*80}")
        
        example = random.choice(trainset)
        
        print(f"\nProblem: {example['problem']}")
        print(f"Expected Answer: {example['answer']}")
        
        try:
            result = model_instance.forward(example['problem'])
            
            if isinstance(result, ExceptionNode):
                print(f"Execution Error: {result.data}")
                feedback = f"Execution error: {result.data}. Please fix the prompts or code to avoid this error."
                score = 0.0
            else:
                feedback, score = generate_feedback(example, result.get('answer', ''), result)
                print(f"Predicted Answer: {result.get('answer', '')}")
                print(f"Score: {score}")
            
            print(f"Feedback: {feedback[:100]}...")
            
            if score < 1.0:
                optimizer.zero_feedback()
                optimizer.backward(result.get('answer'), feedback)
                optimizer.step()
                
                if hasattr(optimizer, 'log') and optimizer.log:
                    last_log = optimizer.log[-1]
                    print(f"\n--- Optimizer Prompt & Response ---")
                    print(f"Response:\n{last_log.get('response', '')}")
                    print("-------------------------------------\n")

                print("✓ Parameters updated:")
                for name, param in model_instance.parameters_dict().items():
                    if getattr(param, 'trainable', False):
                         print(f"  {name}: {param.data}")
            else:
                print("✓ Answer correct, no update needed")
                
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            continue

    return model_instance

def evaluate_baseline(model_instance=None, testset=None, num_threads=20, baseline_model=None):
    """Evaluate unoptimized model on test set in parallel"""
    import concurrent.futures
    from tqdm import tqdm
    
    print(f"\n{'='*80}")
    print(f"Running Parallel Evaluation on {len(testset)} examples ({num_threads} threads)")
    print(f"{'='*80}")
    
    if baseline_model is None:
        baseline_model = AIMEBaseline()
    
    def process_example(example):
        try:
            result = baseline_model.forward(example['problem'])
            score = eval_metric_wrapper(example, result.get('answer', ''))
            return score
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
            return 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        scores = list(tqdm(executor.map(process_example, testset), total=len(testset)))
        
    correct = sum(1 for s in scores if s >= 1.0)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    accuracy = correct / len(scores) if scores else 0.0
    
    print(f"\n{'='*80}")
    print(f"Baseline Test Results:")
    print(f"  Average Score: {avg_score:.3f}")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(scores)})")
    print(f"  Total Examples: {len(scores)}")
    print(f"{'='*80}")

def main(dataset_mode="lite", seed=None, num_steps=20, run_test_baseline=False):
    print("Loading AIME dataset...")
    
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to: {seed}")
    
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
        evaluate_baseline(testset=testset)
        return
    
    print(f"\nRunning optimization...")
    optimized_model = optimization_function(
        model_instance, 
        trainset=trainset, 
        valset=valset,
        num_steps=num_steps
    )
    
    print("\nSaving optimized prompts...")
    optimized_params = {
        "generate_response_template": optimized_model.generate_response_template.data,
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "aime_optimized_params.json")
    with open(output_path, 'w') as f:
        json.dump(optimized_params, f, indent=2)
    
    print(f"✓ Optimized parameters saved to: {output_path}")
    
    print("\nCreating optimized baseline model for final evaluation (parallel)...")
    optimized_baseline = AIMEBaseline()
    optimized_baseline.generate_response_template = optimized_model.generate_response_template.data
    
    print("\n" + "="*80)
    print(f"Final Parallel Evaluation on Test Set ({len(testset)} examples)")
    print("="*80)
    evaluate_baseline(testset=testset, baseline_model=optimized_baseline)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AIME with Trace Optimization")
    parser.add_argument("--dataset-mode", type=str, default="lite", choices=["lite", "full", "tiny", "test"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--run-test-baseline", action="store_true")
    
    args = parser.parse_args()
    
    main(
        dataset_mode=args.dataset_mode, 
        seed=args.seed, 
        num_steps=args.num_steps,
        run_test_baseline=args.run_test_baseline
    )

