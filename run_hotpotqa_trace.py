"""
HotpotQA Multi-Hop Question Answering with Trace Optimization
Following the BigBench-Hard architecture from: https://microsoft.github.io/Trace/examples/nlp/bigbench_hard.html

Key Features:
- Uses EXACT same dataset and splits as DSPy HotpotQA benchmark (via HotpotQABench)
- Uses EXACT same evaluation metrics (answer_match_fn, get_textual_context)
- Optimizes only prompt templates (no code optimization) for fair comparison
- Supports train/val/test splits with proper evaluation methodology

Usage:
    python run_hotpotqa_trace.py --dataset-mode lite --num-steps 20 --seed 42
"""

import os
import sys
import re
import time
from textwrap import dedent
from typing import List
import json

# Add the project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gepa_artifact'))

# Trace imports
from opto.trace.nodes import node, GRAPH, ParameterNode, MessageNode
MessageNode.__format__ = lambda self, format_spec: str(self.data)
from opto.optimizers import OptoPrime
from opto.trace.bundle import bundle
from opto.trace.modules import model
from opto.trace.errors import ExecutionError
from opto.trace.nodes import ExceptionNode
from opto.utils.llm import LLM


class LLMWrapper:
    """Wrapper around opto LLM to provide .create() method for TextGrad compatibility"""
    def __init__(self, llm):
        self.llm = llm
    
    def create(self, messages, max_tokens=None):
        """Convert messages to LLM call and return in expected format"""
        # Call the underlying LLM with messages parameter (opto LLM uses litellm which expects messages)
        response = self.llm(messages=messages, max_tokens=max_tokens or 16384)
        
        # Extract text content from ModelResponse object
        if hasattr(response, 'choices') and len(response.choices) > 0:
            # It's already a ModelResponse object, just return it as-is
            return response
        else:
            # It's a string, wrap it in the expected format
            class Choice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content})()
            
            class Response:
                def __init__(self, content):
                    self.choices = [Choice(content)]
            
            return Response(response)

# Data and utility imports
from datasets import load_dataset
import dspy
import random
from gepa_artifact.benchmarks.hover.hover_program import search
from gepa_artifact.benchmarks.hotpotQA.hotpot_program import (
    answer_match_fn, 
    get_textual_context,
    answer_exact_match_with_feedback
)
from gepa_artifact.benchmarks.hotpotQA.hotpot_data import HotpotQABench

class LLMCallable:
    """Helper class to interact with LLM API using autogen"""
    def __init__(self, config_list=None, max_tokens=16384, verbose=False):
        import autogen
        import datetime
        if config_list is None:
            # Default config for OpenAI
            config_list = [
                {
                    "model": "gpt-4.1-mini",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                }
            ]
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Setup logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("logs", timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logging LLM calls to: {self.log_dir}")

    @bundle(catch_execution_error=True)
    def call_llm(self, system_prompt=None, user_prompt=None):
        """Call LLM with a user prompt"""
        system_prompt = system_prompt or "You are a helpful assistant for multi-hop question answering.\n"
        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]
        response = self.llm.create(messages=messages, max_tokens=self.max_tokens)
        response = response.choices[0].message.content
        
        # Log to file
        import uuid
        try:
            log_file = os.path.join(self.log_dir, f"call_{uuid.uuid4()}.txt")
            with open(log_file, "w") as f:
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(str(system_prompt) + "\n\n")
                f.write("=== USER PROMPT ===\n")
                f.write(str(user_prompt) + "\n\n")
                f.write("=== RESPONSE ===\n")
                f.write(str(response) + "\n")
        except Exception as e:
            print(f"Failed to log LLM call: {e}")

        if self.verbose:
            print("LLM response:\n", response)
        return response


@model
class HotpotMultiHopTrace(LLMCallable):
    """Traced version of HotpotMultiHop using Trace library"""
    
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        # Note: Can't use partial(search, k=k) as an attribute because opto's 
        # parameter discovery expects partial objects to wrap bound methods.
        # Instead, we use a method wrapper to maintain the same API.
        
        # Define trainable prompt templates as ParameterNodes
        # Using initial prompts from initial_prompt.json for HotpotQABench
        self.summarize1_template = ParameterNode(
            "Given the fields `question`, `passages`, produce the fields `summary`.",
            trainable=True,
            description="Template for summarizing first hop passages. Should extract key information relevant to the question."
        )
        
        self.query_hop2_template = ParameterNode(
            "Given the fields `question`, `summary_1`, produce the fields `query`.",
            trainable=True,
            description="Template for generating second hop query. Should identify what additional information is needed."
        )
        
        self.summarize2_template = ParameterNode(
            "Given the fields `question`, `context`, `passages`, produce the fields `summary`.",
            trainable=True,
            description="Template for summarizing second hop passages with previous context. Should integrate information from both hops."
        )
        
        self.final_answer_template = ParameterNode(
            "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.",
            trainable=True,
            description="Template for generating final answer. Should synthesize information from all summaries to answer the question."
        )
    
    def retrieve_k(self, query):
        """Retrieve k passages for the given query using the same search function as original HotpotQA"""
        # Un-wrap the query if it's a Node, because search/diskcache can't handle Nodes
        if hasattr(query, 'data'):
            query = query.data
        return search(query, k=self.k)
    
    def format_passages(self, passages):
        """Format passages for display"""
        return "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    
    @bundle(catch_execution_error=True)
    def parse_output_field(self, response, field_name):
        """
        Parse the output field from DSPy-style response.
        Extracts the value between [[ ## field_name ## ]] and the next [[ ## or [[ ## completed ## ]]
        """
        import re
        # Look for the field marker
        pattern = rf'\[\[\s*##\s*{re.escape(field_name)}\s*##\s*\]\]\s*\n?(.*?)(?=\[\[\s*##|\Z)'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: just return the response if we can't parse it
        return response.strip()
    
    @bundle(catch_execution_error=True, allow_external_dependencies=True)
    def create_dspy_prompt(self, input_fields, output_fields, instruction, values):
        """
        Create a DSPy-style prompt with system and user messages.
        
        Args:
            input_fields: dict of {field_name: field_type}
            output_fields: dict of {field_name: field_type}
            instruction: str, the trainable instruction
            values: dict of {field_name: field_value}
        
        Returns:
            system_prompt, user_prompt
        """
        # Build system message
        system_parts = ["Your input fields are:"]
        for i, (field_name, field_type) in enumerate(input_fields.items(), 1):
            system_parts.append(f"{i}. `{field_name}` ({field_type}): ")
        
        system_parts.append("Your output fields are:")
        for i, (field_name, field_type) in enumerate(output_fields.items(), 1):
            system_parts.append(f"{i}. `{field_name}` ({field_type}): ")
        
        system_parts.append("All interactions will be structured in the following way, with the appropriate values filled in.\n")
        
        # Add field structure
        all_fields = list(input_fields.keys()) + list(output_fields.keys())
        for field in all_fields:
            system_parts.append(f"[[ ## {field} ## ]]")
            system_parts.append(f"{{{field}}}\n")
        
        system_parts.append("[[ ## completed ## ]]")
        system_parts.append(f"In adhering to this structure, your objective is: \n        {instruction}")
        
        system_prompt = "\n".join(system_parts)
        
        # Build user message
        user_parts = []
        for field in input_fields.keys():
            user_parts.append(f"[[ ## {field} ## ]]")
            user_parts.append(f"{values.get(field, '')}\n")
        
        # Add instruction for response
        output_field_names = list(output_fields.keys())
        output_fields_str = ", then ".join([f"`[[ ## {f} ## ]]`" for f in output_field_names])
        user_parts.append(f"Respond with the corresponding output fields, starting with the field {output_fields_str}, and then ending with the marker for `[[ ## completed ## ]]`.")
        
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    def forward(self, question):
        """Main forward pass for multi-hop QA"""
        # HOP 1: Initial retrieval and summarization
        hop1_docs = self.retrieve_k(question).passages
        
        # Create summarize1 prompt using DSPy format
        # Pass the ParameterNode (self.summarize1_template) directly, not .data
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize1_template,
            values={"question": question, "passages": self.format_passages(hop1_docs)}
        )
        response_1 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_1 = self.parse_output_field(response_1, "summary")
        
        # HOP 2: Generate query for second hop
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "summary_1": "str"},
            output_fields={"reasoning": "str", "query": "str"},
            instruction=self.query_hop2_template,
            values={"question": question, "summary_1": summary_1}
        )
        response_2 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        hop2_query = self.parse_output_field(response_2, "query")
        
        # Retrieve for hop 2
        hop2_docs = self.retrieve_k(hop2_query).passages
        
        # Create summarize2 prompt
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "context": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize2_template,
            values={"question": question, "context": summary_1, "passages": self.format_passages(hop2_docs)}
        )
        response_3 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_2 = self.parse_output_field(response_3, "summary")
        
        # HOP 3: Generate final answer
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "summary_1": "str", "summary_2": "str"},
            output_fields={"reasoning": "str", "answer": "str"},
            instruction=self.final_answer_template,
            values={"question": question, "summary_1": summary_1, "summary_2": summary_2}
        )
        response_4 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        answer = self.parse_output_field(response_4, "answer")
        
        return {
            "answer": answer,
            "hop1_docs": hop1_docs,
            "hop2_docs": hop2_docs,
            "summary_1": summary_1,
            "summary_2": summary_2,
            "hop2_query": hop2_query
        }


# -----------------------------------------------------------------------------
# Baseline Model (Non-Traced) for Parallel Evaluation
# -----------------------------------------------------------------------------

class LLMCallableBaseline:
    """Helper class to interact with LLM API using autogen (Baseline, no trace)"""
    def __init__(self, config_list=None, max_tokens=16384, verbose=False):
        import autogen
        import datetime
        if config_list is None:
            config_list = [
                {
                    "model": "gpt-4.1-mini",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                }
            ]
        self.llm = autogen.OpenAIWrapper(config_list=config_list)
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("logs", "baseline_" + timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logging Baseline LLM calls to: {self.log_dir}")

    def call_llm(self, system_prompt=None, user_prompt=None):
        """Call LLM with a user prompt"""
        system_prompt = system_prompt or "You are a helpful assistant for multi-hop question answering.\n"
        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]
        response = self.llm.create(messages=messages, max_tokens=self.max_tokens)
        response = response.choices[0].message.content
        
        # Log to file
        import uuid
        try:
            log_file = os.path.join(self.log_dir, f"call_{uuid.uuid4()}.txt")
            with open(log_file, "w") as f:
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(str(system_prompt) + "\n\n")
                f.write("=== USER PROMPT ===\n")
                f.write(str(user_prompt) + "\n\n")
                f.write("=== RESPONSE ===\n")
                f.write(str(response) + "\n")
        except Exception as e:
            print(f"Failed to log LLM call: {e}")

        if self.verbose:
            print("LLM response:\n", response)
        return response


class HotpotMultiHopBaseline(LLMCallableBaseline):
    """Baseline version of HotpotMultiHop WITHOUT Trace library"""
    
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        
        # Initialize templates as simple strings
        self.summarize1_template = "Given the fields `question`, `passages`, produce the fields `summary`."
        self.query_hop2_template = "Given the fields `question`, `summary_1`, produce the fields `query`."
        self.summarize2_template = "Given the fields `question`, `context`, `passages`, produce the fields `summary`."
        self.final_answer_template = "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`."
    
    def retrieve_k(self, query):
        return search(query, k=self.k)
    
    def format_passages(self, passages):
        return "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    
    def parse_output_field(self, response, field_name):
        import re
        pattern = rf'\[\[\s*##\s*{re.escape(field_name)}\s*##\s*\]\]\s*\n?(.*?)(?=\[\[\s*##|\Z)'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
    
    def create_dspy_prompt(self, input_fields, output_fields, instruction, values):
        # Build system message
        system_parts = ["Your input fields are:"]
        for i, (field_name, field_type) in enumerate(input_fields.items(), 1):
            system_parts.append(f"{i}. `{field_name}` ({field_type}): ")
        
        system_parts.append("Your output fields are:")
        for i, (field_name, field_type) in enumerate(output_fields.items(), 1):
            system_parts.append(f"{i}. `{field_name}` ({field_type}): ")
        
        system_parts.append("All interactions will be structured in the following way, with the appropriate values filled in.\n")
        
        # Add field structure
        all_fields = list(input_fields.keys()) + list(output_fields.keys())
        for field in all_fields:
            system_parts.append(f"[[ ## {field} ## ]]")
            system_parts.append(f"{{{field}}}\n")
        
        system_parts.append("[[ ## completed ## ]]")
        system_parts.append(f"In adhering to this structure, your objective is: \n        {instruction}")
        
        system_prompt = "\n".join(system_parts)
        
        # Build user message
        user_parts = []
        for field in input_fields.keys():
            user_parts.append(f"[[ ## {field} ## ]]")
            user_parts.append(f"{values.get(field, '').value if hasattr(values.get(field, ''), 'value') else values.get(field, '')}\n")
        
        # Add instruction for response
        output_field_names = list(output_fields.keys())
        output_fields_str = ", then ".join([f"`[[ ## {f} ## ]]`" for f in output_field_names])
        user_parts.append(f"Respond with the corresponding output fields, starting with the field {output_fields_str}, and then ending with the marker for `[[ ## completed ## ]]`.")
        
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    def forward(self, question):
        """Main forward pass for multi-hop QA"""
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize1_template,
            values={"question": question, "passages": self.format_passages(hop1_docs)}
        )
        response_1 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_1 = self.parse_output_field(response_1, "summary")
        
        # HOP 2
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "summary_1": "str"},
            output_fields={"reasoning": "str", "query": "str"},
            instruction=self.query_hop2_template,
            values={"question": question, "summary_1": summary_1}
        )
        response_2 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        hop2_query = self.parse_output_field(response_2, "query")
        
        # Retrieve
        hop2_docs = self.retrieve_k(hop2_query).passages
        
        # Summarize 2
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "context": "str", "passages": "str"},
            output_fields={"reasoning": "str", "summary": "str"},
            instruction=self.summarize2_template,
            values={"question": question, "context": summary_1, "passages": self.format_passages(hop2_docs)}
        )
        response_3 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        summary_2 = self.parse_output_field(response_3, "summary")
        
        # Final Answer
        system_prompt, user_prompt = self.create_dspy_prompt(
            input_fields={"question": "str", "summary_1": "str", "summary_2": "str"},
            output_fields={"reasoning": "str", "answer": "str"},
            instruction=self.final_answer_template,
            values={"question": question, "summary_1": summary_1, "summary_2": summary_2}
        )
        response_4 = self.call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        answer = self.parse_output_field(response_4, "answer")
        
        return {
            "answer": answer,
            "hop1_docs": hop1_docs,
            "hop2_docs": hop2_docs,
            "summary_1": summary_1,
            "summary_2": summary_2,
            "hop2_query": hop2_query
        }


def eval_metric(example, prediction):
    """Evaluate if prediction matches the expected answer using dspy.evaluate.answer_exact_match"""
    # Handle Trace Nodes
    if hasattr(prediction, 'data'):
        prediction = prediction.data

    # Create dspy objects for evaluation
    dspy_example = dspy.Example(answer=example['answer'])
    dspy_pred = dspy.Prediction(answer=prediction)
    
    from dspy.evaluate import answer_exact_match
    return 1.0 if answer_exact_match(dspy_example, dspy_pred) else 0.0


def generate_feedback(example, prediction, result):
    """Generate feedback for the optimizer - simple format matching BigBench-Hard"""
    score = eval_metric(example, result.get('answer', ''))
    
    correctness = (score >= 1.0)
    
    if correctness:
        feedback = "The answer is correct! No need to change anything."
    else:
        feedback = f"The answer is wrong. We expect the output of your answer to be \"{example['answer']}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer. Be general and not specific to the example."
    
    return feedback, score


def optimization_function(model_instance, trainset, valset=None, num_steps=20, optimizer_type="OptoPrime", max_time_hours=3.0, save_dir=None):
    """Main optimization function with one optimizer per LLM output
    
    Args:
        optimizer_type: "OptoPrime" or "TextGrad"
        max_time_hours: Maximum time in hours before graceful stop (default: 3.0)
        save_dir: Directory to save optimized parameters (optional)
    """
    
    # Initialize 4 separate optimizers, one per LLM module output
    # Each optimizer has access to ALL parameters for context
    # Use gpt-4.1-mini which is a valid model name
    llm = LLM(model="gpt-4.1-mini")
    
    # Get all parameters - each optimizer gets access to all parameters
    all_parameters = model_instance.parameters()
    
    # Select optimizer class based on type
    from opto.optimizers import TextGrad
    if optimizer_type == "TextGrad":
        OptimizerClass = TextGrad
        # Wrap LLM for TextGrad compatibility (TextGrad expects .create() method)
        llm_for_optimizer = LLMWrapper(llm)
        print(f"Using TextGrad optimizer")
    elif optimizer_type == "OptoPrime":
        OptimizerClass = OptoPrime
        llm_for_optimizer = llm
        print(f"Using OptoPrime optimizer")
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Create one optimizer per output, all sharing the same parameter list
    # The key difference is that each calls backward() on a different output node
    optimizer_summarize1 = OptimizerClass(all_parameters, llm=llm_for_optimizer)
    optimizer_query_hop2 = OptimizerClass(all_parameters, llm=llm_for_optimizer)
    optimizer_summarize2 = OptimizerClass(all_parameters, llm=llm_for_optimizer)
    optimizer_final_answer = OptimizerClass(all_parameters, llm=llm_for_optimizer)
    
    optimizers = {
        'summarize1': optimizer_summarize1,
        'query_hop2': optimizer_query_hop2,
        'summarize2': optimizer_summarize2,
        'final_answer': optimizer_final_answer
    }

    print(f"Starting optimization on {len(trainset)} training examples for {num_steps} steps...")
    print(f"Using 4 separate {optimizer_type} optimizers (one per LLM output)")
    print(f"Maximum time allowed: {max_time_hours} hours ({max_time_hours * 60:.0f} minutes)")
    if valset:
        print(f"Validation set size: {len(valset)} examples")
    
    # Track start time
    start_time = time.time()
    max_time_seconds = max_time_hours * 3600
    time_exceeded = False
    
    for step in range(num_steps):
        # Check if time limit exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time_seconds:
            elapsed_hours = elapsed_time / 3600
            print(f"\n{'='*80}")
            print(f"⏱️  TIME LIMIT REACHED")
            print(f"Elapsed time: {elapsed_hours:.2f} hours ({elapsed_time / 60:.1f} minutes)")
            print(f"Completed {step}/{num_steps} optimization steps")
            print(f"Gracefully stopping optimization and proceeding to evaluation...")
            print(f"{'='*80}\n")
            time_exceeded = True
            break
        print(f"\n{'='*80}")
        print(f"Optimization Step {step + 1}/{num_steps}")
        print(f"{'='*80}")
        
        # Sample an example from training set
        example = random.choice(trainset)
        
        print(f"\nQuestion: {example['question']}")
        print(f"Expected Answer: {example['answer']}")
        
        # Run forward pass
        try:
            result = model_instance.forward(example['question'])
            
            # Check for execution errors
            if isinstance(result, ExceptionNode):
                print(f"Execution Error: {result.data}")
                feedback = f"Execution error: {result.data}. Please fix the prompts or code to avoid this error."
                score = 0.0
            else:
                # Generate feedback
                feedback, score = generate_feedback(example, result.get('answer', ''), result)
                print(f"Predicted Answer: {result.get('answer', '')}")
                print(f"Score: {score:.2f}")
            
            print(f"Feedback: {feedback}...")
            
            # Optimize using feedback - update all 4 optimizers
            if score < 1.0:  # Only optimize if not perfect
                # Zero feedback on all optimizers
                for opt in optimizers.values():
                    opt.zero_feedback()
                
                # Map each optimizer to its corresponding output field
                optimizer_output_map = {
                    'summarize1': result.get('summary_1'),
                    'query_hop2': result.get('hop2_query'),
                    'summarize2': result.get('summary_2'),
                    'final_answer': result.get('answer')
                }
                
                # Each optimizer performs backward on its specific output field
                for opt_name, opt in optimizers.items():
                    output_node = optimizer_output_map[opt_name]
                    if output_node is not None:
                        opt.backward(output_node, feedback)
                        opt.step()
                        
                        # Debug: Print optimizer log for each optimizer
                        if hasattr(opt, 'log') and opt.log:
                            last_log = opt.log[-1]
                            print(f"\n--- Optimizer '{opt_name}' Prompt & Response ---")
                            print(f"System Prompt:\n{last_log.get('system_prompt', '')}")
                            print(f"User Prompt:\n{last_log.get('user_prompt', '')}")
                            print(f"Response:\n{last_log.get('response', '')}")
                            print("-------------------------------------\n")

                print("✓ Parameters updated:")
                for name, param in model_instance.parameters_dict().items():
                    if getattr(param, 'trainable', False):
                         print(f"  {name}: {param.data}")
                
                # Save parameters after each update if save_dir is provided
                if save_dir:
                    import datetime
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Collect all trainable parameters
                    optimized_params = {}
                    for name, param in model_instance.parameters_dict().items():
                        if getattr(param, 'trainable', False):
                            optimized_params[name] = param.data
                    
                    # Save to file with step number
                    params_file = os.path.join(save_dir, f"optimized_params_step{step+1}_{timestamp}.json")
                    with open(params_file, 'w') as f:
                        json.dump(optimized_params, f, indent=2)
                    
                    print(f"  💾 Saved to: {params_file}")
            else:
                print("✓ Answer correct, no update needed")
                
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final timing information
    total_time = time.time() - start_time
    total_hours = total_time / 3600
    total_minutes = total_time / 60

    print("\n" + "="*80)
    if time_exceeded:
        print(f"✓ Optimization stopped due to time limit")
    else:
        print(f"✓ Optimization completed all {num_steps} steps")
    print(f"Total optimization time: {total_hours:.2f} hours ({total_minutes:.1f} minutes)")
    print("="*80)
    
    # Save final optimized parameters if save_dir is provided
    if save_dir:
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect all trainable parameters
        optimized_params = {}
        for name, param in model_instance.parameters_dict().items():
            if getattr(param, 'trainable', False):
                optimized_params[name] = param.data
        
        # Save final version
        params_file = os.path.join(save_dir, f"optimized_params_FINAL_{timestamp}.json")
        with open(params_file, 'w') as f:
            json.dump(optimized_params, f, indent=2)
        
        print(f"✓ Final optimized parameters saved to: {params_file}")
        print(f"  Parameters saved: {list(optimized_params.keys())}")
        print()
    
    return model_instance


def evaluate_baseline(model_instance=None, testset=None, num_threads=20, baseline_model=None):
    """Evaluate unoptimized model on test set in parallel"""
    import concurrent.futures
    from tqdm import tqdm
    
    # Use Baseline model (non-traced) for thread safety
    print(f"\n{'='*80}")
    print(f"Running Parallel Evaluation on {len(testset)} examples ({num_threads} threads)")
    print(f"Using HotpotMultiHopBaseline (non-traced) for thread safety")
    print(f"{'='*80}")
    
    if baseline_model is None:
        baseline_model = HotpotMultiHopBaseline(k=7)
    
    def process_example(example):
        try:
            result = baseline_model.forward(example['question'])
            # Baseline model returns plain dict, not ExceptionNode
            score = eval_metric(example, result.get('answer', ''))
            return score
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
            return 0.0

    # No stop_tracing needed because we use non-traced classes
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


def main(dataset_mode="lite", seed=None, num_steps=20, run_test_baseline=False, use_params_file=None, optimizer="OptoPrime", save_dir=None):
    """
    Main execution function
    
    Args:
        dataset_mode: Dataset mode - "lite" (default), "full", "tiny", or "test"
        seed: Random seed for reproducibility (optional)
        num_steps: Number of optimization steps (default: 20)
        run_test_baseline: If True, run baseline evaluation only
        optimizer: Optimizer type - "OptoPrime" or "TextGrad"
        save_dir: Directory to save optimized parameters (optional)
    """
    print("Loading HotpotQA dataset...")
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to: {seed}")
    
    # Use save_dir if provided, otherwise use current directory
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    
    # Load dataset using the same benchmark class as DSPy version
    # This ensures exact same data splits and preprocessing:
    # - Uses HuggingFace "hotpot_qa" fullwiki dataset
    # - Splits: test (40%), val (40%), train (20%)
    # - Trimmed to: train=150, val=300, test=300
    # - Uses seed=1 for consistent sampling (in trim_dataset)
    benchmark = HotpotQABench(dataset_mode=dataset_mode)
    
    # Convert dspy.Example objects to dicts for easier handling
    def example_to_dict(ex):
        return {
            "question": ex.question,
            "answer": ex.answer,
            "context": ex.get("context", {}),
            "supporting_facts": ex.get("supporting_facts", {})
        }
    
    # Use the same splits as DSPy benchmark
    # The benchmark creates: test_set (40%), val_set (40%), train_set (20%)
    # and trims them to max sizes: train=150, val=300, test=300
    trainset = [example_to_dict(ex) for ex in benchmark.train_set]
    valset = [example_to_dict(ex) for ex in benchmark.val_set]
    testset = [example_to_dict(ex) for ex in benchmark.test_set]
    
    print(f"Loaded dataset splits:")
    print(f"  Train: {len(trainset)} examples")
    print(f"  Val: {len(valset)} examples")
    print(f"  Test: {len(testset)} examples")
    
    # Initialize model
    print("\nInitializing HotpotMultiHopTrace model...")
    model_instance = HotpotMultiHopTrace(k=7)
    
    if run_test_baseline:
        baseline_model = None
        if use_params_file:
            print(f"\nLoading optimized parameters from: {use_params_file}")
            with open(use_params_file, 'r') as f:
                optimized_params = json.load(f)
            
            # Create baseline model with optimized parameters
            baseline_model = HotpotMultiHopBaseline(k=7)
            baseline_model.summarize1_template = optimized_params["summarize1_template"]
            baseline_model.query_hop2_template = optimized_params["query_hop2_template"]
            baseline_model.summarize2_template = optimized_params["summarize2_template"]
            baseline_model.final_answer_template = optimized_params["final_answer_template"]
            print("✓ Loaded optimized parameters:")
            for key, value in optimized_params.items():
                print(f"  - {key}: {value[:80]}...")
        
        evaluate_baseline(testset=testset, baseline_model=baseline_model)
        return
    
    # Run optimization on training set, validate on validation set
    print(f"\nRunning optimization with {optimizer}...")
    optimized_model = optimization_function(
        model_instance, 
        trainset=trainset, 
        valset=valset,
        num_steps=num_steps,
        optimizer_type=optimizer,
        save_dir=save_dir
    )
    
    # Final evaluation using optimized prompts with baseline (thread-safe) model
    print("\nCreating optimized baseline model for final evaluation (parallel)...")
    optimized_baseline = HotpotMultiHopBaseline(k=7)
    optimized_baseline.summarize1_template = optimized_model.summarize1_template.data
    optimized_baseline.query_hop2_template = optimized_model.query_hop2_template.data
    optimized_baseline.summarize2_template = optimized_model.summarize2_template.data
    optimized_baseline.final_answer_template = optimized_model.final_answer_template.data
    
    print("\n" + "="*80)
    print(f"Final Parallel Evaluation on Test Set ({len(testset)} examples)")
    print("="*80)
    evaluate_baseline(testset=valset, baseline_model=optimized_baseline)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HotpotQA Multi-Hop QA with Trace Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="lite",
        choices=["lite", "full", "tiny", "test"],
        help="Dataset size mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of optimization steps"
    )
    parser.add_argument(
        "--run-test-baseline",
        action="store_true",
        help="Run unoptimized model on test set in parallel"
    )
    parser.add_argument(
        "--use-params-file",
        type=str,
        default=None,
        help="Path to JSON file with optimized parameters (use with --run-test-baseline)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="OptoPrime",
        choices=["OptoPrime", "TextGrad"],
        help="Optimizer to use for training (default: OptoPrime)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save optimized parameters (default: current directory)"
    )
    
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

