import dspy
from dspy.teleprompt.bootstrap_finetune import FailedPrediction

class GenerateKBQueries(dspy.Signature):
    """Generate targeted knowledge base queries to improve model performance on a specific task. 
    
For the specific task, you are provided with a set of example inputs and the model's incorrect outputs, along with feedback explaining the issues with the outputs. Your task is to create targetted queries that retrieve information from a knowledge base to address specific feedback on model outputs. The goal is to find as much information from the knowledge base as possible to help the model improve its performance on the task."""
    task_samples: str = dspy.InputField()
    kb_queries: list[str] = dspy.OutputField()

class FetchFromKB(dspy.Module):
    def __init__(self, query_engine):
        # self.knowledge_index = knowledge_index
        self.query_engine = query_engine # knowledge_index.as_query_engine()
        # self.retriever = knowledge_index.as_retriever()
        self.gen_kb_queries = dspy.ChainOfThought(GenerateKBQueries)
    
    def forward(self, samples_to_search_for):
        o = self.gen_kb_queries(task_samples=samples_to_search_for)
        queries = o.kb_queries
        queries = queries[:min(5, len(queries))]
        responses = []
        for query in queries:
            query = "Answer the following query in detail: " + query + "\n\nInclude as much relevant information from the documents as possible."
            response = self.query_engine.query(query).response
            responses.append(response)
        
        return dspy.Prediction(queries=queries, responses=responses)

prompt1 = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

prompt2 = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Some additional information about the task from reference materials:
```
<reference_materials>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

def call_lm_and_extract_response(prompt, lm, current_instruction_doc, user_examples_and_feedback, reference_materials=None):
    full_prompt = prompt.replace("<curr_instructions>", current_instruction_doc)
    full_prompt = full_prompt.replace("<inputs_outputs_feedback>", user_examples_and_feedback)
    if reference_materials is not None:
        full_prompt = full_prompt.replace("<reference_materials>", reference_materials)
    
    lm_out = lm(full_prompt, max_tokens=16384)[0].strip()
    # Extract ``` blocks
    if lm_out.count("```") >= 2:
        start = lm_out.find("```")
        end = lm_out.rfind("```")
        if start >= end:
            return lm_out, full_prompt
        if start == -1 or end == -1:
            return lm_out, full_prompt
        else:
            return lm_out[start+3:end].strip(), full_prompt
    else:
        lm_out = lm_out.strip()
        if lm_out.startswith("```"):
            lm_out = lm_out[3:]
        if lm_out.endswith("```"):
            lm_out = lm_out[:-3]
        return lm_out, full_prompt

class ProposeNewInstructionModule:
    def __init__(self, base_program, instruction_lm, dataset_with_feedback, knowledgebase_qe, kb_fetch_lm=None):
        self.base_program = base_program
        self.dataset_with_feedback = dataset_with_feedback

        signature_to_use = prompt1 # ImproveInstruction
        if knowledgebase_qe is not None:
            # signature_to_use = signature_to_use.append(
            #     "reference_materials",
            #     dspy.InputField(
            #         desc="Additional domain knowledge, context, specifications, or best practices that should inform the improved instructions.",
            #         format=str
            #     )
            # )
            signature_to_use = prompt2
            self.kb_query = knowledgebase_qe
            self.kb_fetch = FetchFromKB(knowledgebase_qe)
            if kb_fetch_lm is not None:
                self.kb_fetch.set_lm(kb_fetch_lm)
        else:
            self.kb_query = None
            self.kb_fetch = None

        # self.instruction_propose_module = dspy.Predict(signature_to_use)
        self.instruction_lm = instruction_lm
    
    def format_samples(self, samples):
        def convert_sample_to_markdown(sample, examplenum):
            s = "# Example " + str(examplenum) + "\n"
            s += "## Inputs\n"
            contains_history = False
            history_key_name = None
            if 'inputs' not in sample:
                raise ValueError("Sample does not contain 'inputs' key." + str(sample))
            for input_key, input_val in sample['inputs'].items():
                if isinstance(input_val, dspy.History):
                    contains_history = True
                    assert history_key_name is None
                    history_key_name = input_key
            
            if contains_history:
                s += f"### Context\n"
                s += "```json\n"
                for i, message in enumerate(sample['inputs'][history_key_name].messages):
                    s += f"  {i}: {message}\n"
                s += "```\n\n"
            
            for input_key, input_val in sample['inputs'].items():
                if contains_history and input_key == history_key_name:
                    continue
                s += f"### {input_key}\n"
                s += str(input_val) + "\n\n"

            s += "## Generated Outputs\n"
            if isinstance(sample['generated_output'], FailedPrediction):
                print("Adding raw response for failed prediction.")
                s += "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                s += "```\n"
                s += sample['generated_output'].completion_text + "\n"
                s += "```\n\n"
            else:
                for output_key, output_val in sample['generated_output'].items():
                    s += f"### {output_key}\n"
                    s += str(output_val) + "\n\n"

            s += "## Feedback\n"
            s += sample['feedback'] + "\n\n"

            return s
        
        return "\n\n".join([convert_sample_to_markdown(sample, i+1) for i, sample in enumerate(samples)])

    def fetch_relevant_information(self, samples):
        qe_response = self.kb_fetch(samples_to_search_for=self.format_samples(samples))
        response = "\n".join(qe_response.responses)
        return response
    
    def compile(self):
        instruction = self.base_program.signature.instructions

        sample = self.dataset_with_feedback
        formatted_samples = self.format_samples(sample)

        if self.kb_query:
            kb_info = self.fetch_relevant_information(sample)
            # module_output = self.instruction_propose_module(
            #     current_instruction_doc=instruction,
            #     user_examples_and_feedback=self.format_samples(sample),
            #     reference_materials=kb_info
            # )
            new_instruction, full_prompt = call_lm_and_extract_response(
                prompt2,
                self.instruction_lm,
                current_instruction_doc=instruction,
                user_examples_and_feedback=formatted_samples,
                reference_materials=kb_info
            )
        else:
            kb_info = "No QueryEngine Provided"
            # module_output = self.instruction_propose_module(
            #     current_instruction_doc=instruction,
            #     user_examples_and_feedback=self.format_samples(sample),
            # )
            new_instruction, full_prompt = call_lm_and_extract_response(
                prompt1,
                self.instruction_lm,
                current_instruction_doc=instruction,
                user_examples_and_feedback=formatted_samples
            )
            
        # new_instruction = module_output.improved_instruction_doc

        return {
            "new_instruction": new_instruction,
            "module_output": {"improved_instruction_doc": new_instruction},
            "kb_info": kb_info,
            "full_prompt_to_teacher_lm": full_prompt,  # 完整的 prompt 用于计算 cost
            "all_inputs": {
                "current_instruction_doc": instruction,
                "user_examples_and_feedback": formatted_samples,
                "reference_materials": kb_info
            }
        }
