import os
import re
import sys
import random
import time
from opto.trace.bundle import bundle
from opto.trace.nodes import ExceptionNode
from opto.optimizers import OptoPrime, TextGrad
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
        system_prompt = system_prompt or "You are a helpful assistant.\n"
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
        system_prompt = system_prompt or "You are a helpful assistant.\n"
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

@bundle(catch_execution_error=True, allow_external_dependencies=True)
def create_dspy_prompt(input_fields, output_fields, instruction, values):
    """
    Create a DSPy-style prompt with system and user messages.
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

def create_dspy_prompt_baseline(input_fields, output_fields, instruction, values):
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
        # Handle potential Trace nodes in baseline if mixed
        val = values.get(field, '')
        if hasattr(val, 'value'):
             val = val.value
        user_parts.append(f"{val}\n")
    
    # Add instruction for response
    output_field_names = list(output_fields.keys())
    output_fields_str = ", then ".join([f"`[[ ## {f} ## ]]`" for f in output_field_names])
    user_parts.append(f"Respond with the corresponding output fields, starting with the field {output_fields_str}, and then ending with the marker for `[[ ## completed ## ]]`.")
    
    user_prompt = "\n".join(user_parts)
    
    return system_prompt, user_prompt

@bundle(catch_execution_error=True)
def parse_output_field(response, field_name):
    """
    Parse the output field from DSPy-style response.
    """
    import re
    pattern = rf'\[\[\s*##\s*{re.escape(field_name)}\s*##\s*\]\]\s*\n?(.*?)(?=\[\[\s*##|\Z)'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

def parse_output_field_baseline(response, field_name):
    import re
    pattern = rf'\[\[\s*##\s*{re.escape(field_name)}\s*##\s*\]\]\s*\n?(.*?)(?=\[\[\s*##|\Z)'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

def train_optimization_loop(
    model_instance,
    trainset,
    optimizer_config, # Dict[str, str] mapping optimizer_name -> output_field_in_result
    feedback_fn,      # Callable(example, result, full_result) -> (feedback_str, score)
    input_key="question", 
    num_steps=20,
    optimizer_type="OptoPrime",  # "OptoPrime" or "TextGrad"
    max_time_hours=3.0,  # Maximum time in hours before graceful stop
    save_dir=None  # Directory to save optimized parameters
):
    """
    Generic optimization loop for Trace models.
    
    Args:
        model_instance: The Trace model instance.
        trainset: List of training examples (dicts).
        optimizer_config: Dict mapping optimizer names to result keys they should optimize.
        feedback_fn: Function that takes (example, result_field, full_result) and returns (feedback, score).
                     Note: result_field is extracted based on logic, but here we might pass full result usually.
        input_key: Key in example dict to pass to model.forward().
        num_steps: Number of optimization steps.
        optimizer_type: Type of optimizer to use ("OptoPrime" or "TextGrad").
        max_time_hours: Maximum time in hours before graceful stop (default: 3.0).
        save_dir: Directory to save optimized parameters (optional).
    """
    # Initialize optimizers
    llm = LLM(model="gpt-4.1-mini")
    all_params = model_instance.parameters()
    
    # Select optimizer class based on type
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
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Must be 'OptoPrime' or 'TextGrad'")
    
    optimizers = {
        name: OptimizerClass(all_params, llm=llm_for_optimizer) 
        for name in optimizer_config.keys()
    }
    
    print(f"Starting optimization on {len(trainset)} training examples for {num_steps} steps...")
    print(f"Using {len(optimizers)} {optimizer_type} optimizer(s): {list(optimizers.keys())}")
    print(f"Maximum time allowed: {max_time_hours} hours ({max_time_hours * 60:.0f} minutes)")
    
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
        
        example = random.choice(trainset)
        
        print(f"\nInput ({input_key}): {example.get(input_key, 'N/A')}")
        
        try:
            # Forward pass
            result = model_instance.forward(example[input_key])
            
            if isinstance(result, ExceptionNode):
                print(f"Execution Error: {result.data}")
                score = 0.0
                feedback = f"Execution Error: {result.data}. Please fix the prompts or code to avoid this error."
            else:
                # Generate feedback
                # We pass the full result to feedback_fn usually, or handle it there
                feedback, score = feedback_fn(example, result, result)
                print(f"Score: {score}")
            
            print(f"Feedback: {feedback[:200]}...")
            
            if score < 1.0:
                # Zero feedback
                for opt in optimizers.values():
                    opt.zero_feedback()
                
                # Backward pass
                updated = False
                for opt_name, opt in optimizers.items():
                    result_key = optimizer_config[opt_name]
                    output_node = result.get(result_key)
                    
                    if output_node is not None and output_node != "":
                        opt.backward(output_node, feedback)
                        opt.step()
                        updated = True
                        
                        # Debug log for optimizer (optional)
                        if hasattr(opt, 'log') and opt.log:
                            last_log = opt.log[-1]
                            print(f"\n--- Optimizer '{opt_name}' Prompt & Response ---")
                            print(f"System Prompt:\n{last_log.get('system_prompt', '')}")
                            print(f"User Prompt:\n{last_log.get('user_prompt', '')}")
                            print(f"Response:\n{last_log.get('response', '')}")
                            print("-------------------------------------\n")
                
                if updated:
                    print("✓ Parameters updated")
                    for name, param in model_instance.parameters_dict().items():
                        if getattr(param, 'trainable', False):
                            # Print first 50 chars of param to verify update
                            print(f"  {name}: {str(param.data)[:50]}...")
                    
                    # Save parameters after each update if save_dir is provided
                    if save_dir:
                        import json
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
                    print("⚠ No parameters updated (check output keys)")
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
    
    print(f"\n{'='*80}")
    if time_exceeded:
        print(f"✓ Optimization stopped due to time limit")
    else:
        print(f"✓ Optimization completed all {num_steps} steps")
    print(f"Total optimization time: {total_hours:.2f} hours ({total_minutes:.1f} minutes)")
    print(f"{'='*80}\n")
    
    # Save final optimized parameters if save_dir is provided
    if save_dir:
        import json
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

def evaluate_baseline_concurrent(
    testset,
    forward_fn, # Method or function to call with input
    eval_fn,    # Function(example, result) -> score
    input_key="question",
    num_threads=20
):
    """
    Generic parallel evaluation function.
    """
    import concurrent.futures
    from tqdm import tqdm
    
    print(f"\n{'='*80}")
    print(f"Running Parallel Evaluation on {len(testset)} examples ({num_threads} threads)")
    print(f"{'='*80}")
    
    def process_example(example):
        try:
            result = forward_fn(example[input_key])
            return eval_fn(example, result)
        except Exception as e:
            # print(f"Eval Error: {e}")
            return 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        scores = list(tqdm(executor.map(process_example, testset), total=len(testset)))
        
    correct = sum(1 for s in scores if s >= 1.0)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    accuracy = correct / len(scores) if scores else 0.0
    
    print(f"\n{'='*80}")
    print(f"Test Results:")
    print(f"  Average Score: {avg_score:.3f}")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(scores)})")
    print(f"  Total Examples: {len(scores)}")
    print(f"{'='*80}")
    
    return avg_score
