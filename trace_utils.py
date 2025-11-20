import os
import re
import sys
from opto.trace.bundle import bundle

class LLMCallable:
    """Helper class to interact with LLM API using autogen"""
    def __init__(self, config_list=None, max_tokens=2048, verbose=False):
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
    def __init__(self, config_list=None, max_tokens=2048, verbose=False):
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

