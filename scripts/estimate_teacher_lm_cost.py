#!/usr/bin/env python3
"""
Estimate Claude (Teacher LM) cost from instruction_proposals.jsonl

This script estimates the cost of using Claude as the Teacher LM for GEPA-SGD
based on the instruction proposals generated.
"""

import json
import tiktoken
import argparse
from pathlib import Path


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens using tiktoken.
    Note: tiktoken doesn't have Claude encodings, so we use GPT-4 as a proxy.
    Claude's tokenizer is similar but not identical.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_input_tokens_per_call(instruction_length: int, full_prompt: str = None) -> int:
    """
    Estimate total input tokens for a ProposeNewInstructionModule call.
    
    If full_prompt is provided, use that for accurate counting.
    Otherwise, estimate based on:
    - System prompt / template (~500 tokens)
    - Current instruction (variable)
    - Formatted traces with feedback (~2000-5000 tokens per minibatch of 10)
    - KB info (~50 tokens)
    """
    if full_prompt:
        # Use actual prompt for accurate counting
        template_overhead = 500  # The prompt template itself
        traces_tokens = count_tokens(full_prompt)
        return template_overhead + instruction_length + traces_tokens
    else:
        # Fallback to estimation
        template_overhead = 500
        traces_with_feedback = 3000  # Average for minibatch of 10
        kb_info = 50
        return template_overhead + instruction_length + traces_with_feedback + kb_info


def estimate_cost_from_proposals(jsonl_path: str, verbose: bool = False):
    """
    Estimate Claude cost from instruction_proposals.jsonl
    
    Claude Sonnet 4.5 pricing (as of late 2025):
    - Input: $3 / 1M tokens
    - Output: $15 / 1M tokens
    """
    # Claude Sonnet pricing
    INPUT_COST_PER_MILLION = 3.0  # $3 per 1M input tokens
    OUTPUT_COST_PER_MILLION = 15.0  # $15 per 1M output tokens
    
    proposals = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                proposals.append(json.loads(line))
    
    total_input_tokens = 0
    total_output_tokens = 0
    has_full_prompts = False
    
    for i, proposal in enumerate(proposals):
        old_instruction = proposal.get('old_instruction', '')
        new_instruction = proposal.get('new_instruction', '')
        full_prompt = proposal.get('full_prompt_to_teacher_lm', None)
        
        if full_prompt:
            has_full_prompts = True
        
        # Count output tokens (the new instruction)
        output_tokens = count_tokens(new_instruction)
        total_output_tokens += output_tokens
        
        # Estimate input tokens
        old_instruction_tokens = count_tokens(old_instruction)
        input_tokens = estimate_input_tokens_per_call(old_instruction_tokens, full_prompt)
        total_input_tokens += input_tokens
        
        if verbose:
            print(f"Iteration {proposal.get('iteration')}: "
                  f"input ~{input_tokens}, output {output_tokens}")
    
    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    total_cost = input_cost + output_cost
    
    print("\n" + "="*60)
    print("Claude Teacher LM Cost Estimation")
    print("="*60)
    print(f"Number of iterations: {len(proposals)}")
    if has_full_prompts:
        print(f"Input tokens (ACTUAL): {total_input_tokens:,}")
    else:
        print(f"Estimated input tokens: {total_input_tokens:,}")
    print(f"Output tokens (counted): {total_output_tokens:,}")
    print("\nCost breakdown:")
    print(f"  Input cost:  ${input_cost:.4f}")
    print(f"  Output cost: ${output_cost:.4f}")
    print(f"  Total cost:  ${total_cost:.4f}")
    print("="*60)
    if not has_full_prompts:
        print("\nNote: Input tokens are ESTIMATED. Actual cost depends on")
        print("the length of traces and feedback included in each call.")
        print("Re-run the experiment with the latest code to get exact costs.")
    
    return {
        'num_iterations': len(proposals),
        'estimated_input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost
    }


def main():
    parser = argparse.ArgumentParser(
        description='Estimate Claude (Teacher LM) cost from instruction_proposals.jsonl'
    )
    parser.add_argument(
        'jsonl_path',
        type=str,
        help='Path to instruction_proposals.jsonl file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print per-iteration details'
    )
    
    args = parser.parse_args()
    
    if not Path(args.jsonl_path).exists():
        print(f"Error: File not found: {args.jsonl_path}")
        return 1
    
    estimate_cost_from_proposals(args.jsonl_path, args.verbose)
    return 0


if __name__ == '__main__':
    exit(main())
