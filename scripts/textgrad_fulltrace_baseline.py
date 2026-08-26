#!/usr/bin/env python3
"""
TextGrad-style baseline for full trace prompt optimization.

流程：
1. 每条 full trace → LLM → 生成 gradient（改进建议）
2. 聚合所有 gradients
3. Prompt + Gradients → LLM → New Prompt

用法：
python scripts/textgrad_fulltrace_baseline.py \
    --traces_dir verusage/traces \
    --base_prompts postcondition_repair,induction \
    --output_dir /tmp/textgrad_output \
    --lm_model anthropic/claude-sonnet-4-5-20250929
"""
import argparse
import json
import os
import random
import re
from pathlib import Path


# ============== Prompt Templates ==============

GRADIENT_PROMPT = """You are analyzing an execution trace of a Verus verification assistant.

The assistant was given the following instructions:
```
{curr_instructions}
```

Below is the full execution trace (all sub-agent interactions):
```
{full_trace}
```

The final result of this trace was: {final_result}

Your task is to identify specific issues with the assistant's behavior and suggest concrete improvements to the instructions.

Focus on:
1. What went wrong (if the trace failed)?
2. What reasoning patterns led to the failure?
3. What specific knowledge or guidance was missing from the instructions?

Provide your analysis as a concise "gradient" - a list of specific improvements:
```
GRADIENT:
- [improvement 1]
- [improvement 2]
...
```
"""

APPLY_GRADIENTS_PROMPT = """I provided an assistant with the following instructions to perform a task for me:

<current_instructions>
{curr_instructions}
</current_instructions>

I ran the assistant on {num_traces} different inputs and collected feedback on how to improve the instructions.

Here are the improvement suggestions (gradients) from each trace:

<gradients>
{aggregated_gradients}
</gradients>

Your task is to write improved instructions for the assistant that incorporate these suggestions.

Guidelines:
1. Read all the gradients carefully and identify common patterns
2. Incorporate specific domain knowledge mentioned in the gradients
3. Add concrete guidance for avoiding the identified failure modes
4. Keep the instructions clear and actionable
5. The new instructions should be a complete replacement, not a diff

Output the complete new instructions between <BEGIN> and <END> tags:
"""


# ============== Utility Functions ==============

def extract_code_block(text: str) -> str:
    """Extract content from ``` blocks, preferring the longest non-empty block."""
    if "```" not in text:
        return text.strip()
    
    # Find all code blocks with their content
    blocks = re.findall(r"```(?:\w*\n?)?([\s\S]*?)```", text)
    
    # Filter out empty or very short blocks (like just "</response>")
    valid_blocks = [b.strip() for b in blocks if len(b.strip()) > 20]
    
    if valid_blocks:
        # Return the longest valid block
        return max(valid_blocks, key=len)
    
    # If no valid blocks, try to find content between first ``` and last ```
    first_fence = text.find("```")
    last_fence = text.rfind("```")
    if first_fence != -1 and last_fence != -1 and last_fence > first_fence:
        # Skip the first ``` and any language identifier
        start = first_fence + 3
        # Skip language identifier if present
        newline_after_fence = text.find("\n", start)
        if newline_after_fence != -1 and newline_after_fence < start + 20:
            start = newline_after_fence + 1
        content = text[start:last_fence].strip()
        if len(content) > 20:
            return content
    
    # Fallback: return original text
    return text.strip()


def extract_json_block(text: str) -> dict | None:
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        for blk in blocks:
            blk = blk.strip()
            if blk.startswith("{") and blk.endswith("}"):
                try:
                    return json.loads(blk)
                except Exception:
                    pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def normalize_prompt_name(name: str) -> str:
    return name.replace(".predict", "")


def load_prompt_text(prompt_dir: Path, name: str) -> str | None:
    path = prompt_dir / f"{name}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def build_prompt_map(prompt_dir: Path) -> dict:
    """Build mapping from action names to prompt file names."""
    prompt_map = {}
    try:
        from verusage.agents.actions.merged_prompt_ablation import MergedPromptAblationAction
        for action_type, prompt_name in MergedPromptAblationAction().action_prompt_mapping.items():
            prompt_map[action_type.value] = prompt_name
    except Exception as exc:
        print(f"[warn] Failed to load merged prompt mapping: {exc}")

    # Default mappings
    prompt_map.setdefault("nonlinear_arithmetic", "arithmetic_reasoning")
    prompt_map.setdefault("invariant_front_repair", "invariant_front_repair_general")
    prompt_map.setdefault("postcondition_repair", "postcondition_repair_basic")
    return prompt_map


def resolve_prompt_key(name: str, prompt_map: dict, available_prompts: set) -> str | None:
    mapped = prompt_map.get(name, name)
    if mapped in available_prompts:
        return mapped
    return None


def collect_full_trace(trace_path: Path) -> dict | None:
    """Collect entire trace as a single sample."""
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    segments = []
    
    for key, val in sorted(data.items()):
        if not key.startswith("sub_agent_calling_"):
            continue
        if not isinstance(val, dict):
            continue
        name = val.get("name")
        trace = val.get("trace", [])
        if not name or not trace:
            continue
        segments.append({"name": name, "trace": trace, "key": key})
    
    if not segments:
        return None
    
    # Collect segment names for filtering
    segment_names = set()
    for seg in segments:
        segment_names.add(normalize_prompt_name(seg["name"]))
    
    # Concatenate all messages
    all_messages = []
    for seg in segments:
        for msg in seg["trace"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            all_messages.append(f"[{seg['name']}] {role}: {content}")
    
    full_content = "\n\n".join(all_messages)
    
    # Get final result from last segment
    last_seg = segments[-1]
    last_assistant = next((m for m in reversed(last_seg["trace"]) if m.get("role") == "assistant"), None)
    if last_assistant:
        data = extract_json_block(last_assistant.get("content", ""))
        if data and data.get("accepted"):
            final_result = "SUCCESS (accepted=True)"
            final_score = 1.0
        else:
            final_result = "FAILURE (accepted=False)"
            final_score = 0.0
    else:
        final_result = "UNKNOWN"
        final_score = 0.0
    
    return {
        "trace_path": str(trace_path),
        "full_content": full_content,
        "segment_names": segment_names,
        "final_result": final_result,
        "final_score": final_score,
    }


def summarize_lm_cost(lm) -> tuple[float | None, int, int, int]:
    """Summarize LLM usage cost."""
    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    missing_cost = 0
    for entry in getattr(lm, "history", []):
        cost = entry.get("cost")
        if cost is None:
            missing_cost += 1
        else:
            total_cost += float(cost)
        usage = entry.get("usage", {}) or {}
        total_prompt += int(usage.get("prompt_tokens", 0) or 0)
        total_completion += int(usage.get("completion_tokens", 0) or 0)
    if not getattr(lm, "history", []):
        return None, 0, 0, 0
    return total_cost, total_prompt, total_completion, missing_cost


# ============== TextGrad Core ==============

class TextGradOptimizer:
    def __init__(self, lm, prompt_dir: Path, prompt_map: dict, available_prompts: set):
        self.lm = lm
        self.prompt_dir = prompt_dir
        self.prompt_map = prompt_map
        self.available_prompts = available_prompts
    def _split_trace_content(self, content: str) -> list[str]:
        """Split trace content in half recursively until each chunk fits in context."""
        # Try to send to LLM - if it fails with context error, split in half
        # For now, estimate: 200K tokens limit, ~4 chars/token, leave 20K for prompt = ~720K chars
        max_chars = 720000
        
        if len(content) <= max_chars:
            return [content]
        
        # Split in half recursively
        mid = len(content) // 2
        left = content[:mid].strip()
        right = content[mid:].strip()
        
        # Recursively split if still too big
        return self._split_trace_content(left) + self._split_trace_content(right)
    
    def compute_gradient(self, full_trace: dict, curr_instructions: str) -> list[str]:
        """Compute textual gradients for a trace. Returns list of gradients (one per chunk if split)."""
        trace_content = full_trace["full_content"]
        chunks = self._split_trace_content(trace_content)
        
        if len(chunks) > 1:
            print(f"    [info] Trace split into {len(chunks)} chunks")
        
        gradients = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_label = f" (chunk {chunk_idx+1}/{len(chunks)})" if len(chunks) > 1 else ""
            
            prompt = GRADIENT_PROMPT.format(
                curr_instructions=curr_instructions,
                full_trace=chunk,
                final_result=full_trace["final_result"] + chunk_label,
            )
            
            response = self.lm(prompt, max_tokens=2048)[0]
            gradients.append(response.strip())
        
        return gradients
    
    def apply_gradients(self, gradients: list[str], curr_instructions: str) -> str:
        """Apply aggregated gradients to generate new instructions."""
        # Format gradients
        aggregated = ""
        for i, grad in enumerate(gradients):
            aggregated += f"=== Trace {i+1} ===\n{grad}\n\n"
        
        prompt = APPLY_GRADIENTS_PROMPT.format(
            curr_instructions=curr_instructions,
            num_traces=len(gradients),
            aggregated_gradients=aggregated,
        )
        
        response = self.lm(prompt, max_tokens=4096)[0]
        
        # Extract content between <BEGIN> and <END>
        match = re.search(r"<BEGIN>([\s\S]*?)<END>", response)
        if match:
            return match.group(1).strip()
        
        # Fallback: return full response
        return response.strip()
    
    def optimize(self, traces: list[dict], base_prompt_key: str) -> dict:
        """Run full TextGrad optimization."""
        # Load base prompt
        curr_instructions = load_prompt_text(self.prompt_dir, base_prompt_key)
        if curr_instructions is None:
            return {"error": f"Prompt file not found: {base_prompt_key}"}
        
        # Build a set of action names that map to this prompt
        # This allows multiple prompts to share traces from the same action family
        matching_actions = set()
        for action_name, prompt_key in self.prompt_map.items():
            if prompt_key == base_prompt_key:
                matching_actions.add(action_name)
        # Also check if base_prompt_key itself is a prefix match for segment names
        # e.g., postcondition_repair_basic should match traces with postcondition_repair
        base_action = base_prompt_key.rsplit('_', 1)[0] if '_' in base_prompt_key else base_prompt_key
        matching_actions.add(base_action)
        matching_actions.add(base_prompt_key)
        
        # Filter traces that contain this subagent or related actions
        matching_traces = []
        for trace in traces:
            for seg_name in trace["segment_names"]:
                # Direct match with action name
                if seg_name in matching_actions:
                    matching_traces.append(trace)
                    break
                # Check if segment maps to this prompt
                seg_key = resolve_prompt_key(seg_name, self.prompt_map, self.available_prompts)
                if seg_key == base_prompt_key:
                    matching_traces.append(trace)
                    break
        
        if not matching_traces:
            return {"error": f"No traces contain subagent for '{base_prompt_key}'"}
        
        print(f"[{base_prompt_key}] Computing gradients for {len(matching_traces)} traces...")
        
        # Step 1: Compute gradient for each trace (may return multiple if trace is split)
        gradients = []
        for i, trace in enumerate(matching_traces):
            print(f"  [{i+1}/{len(matching_traces)}] {Path(trace['trace_path']).name}")
            grads = self.compute_gradient(trace, curr_instructions)
            gradients.extend(grads)  # extend since compute_gradient returns a list
        
        # Step 2: Apply gradients
        print(f"[{base_prompt_key}] Applying {len(gradients)} gradients...")
        new_instructions = self.apply_gradients(gradients, curr_instructions)
        
        return {
            "base_prompt_key": base_prompt_key,
            "num_traces": len(matching_traces),
            "num_gradients": len(gradients),
            "new_instructions": new_instructions,
            "gradients": gradients,
        }


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="TextGrad-style full trace prompt optimization.")
    parser.add_argument("--traces_dir", required=True, help="Directory with trace JSON files.")
    parser.add_argument(
        "--prompt_dir",
        default=str(Path(__file__).resolve().parents[1] / "verusage" / "agents" / "prompts"),
        help="Directory containing prompt .md files.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write optimized prompts.")
    parser.add_argument(
        "--base_prompts",
        required=True,
        help="Comma-separated prompt names to optimize (e.g., postcondition_repair,induction).",
    )
    parser.add_argument("--lm_model", required=True, help="LM model name (e.g., anthropic/claude-sonnet-4-5-20250929).")
    parser.add_argument("--lm_api_key_env", default="ANTHROPIC_API_KEY", help="Env var for API key.")
    parser.add_argument("--max_traces", type=int, default=None, help="Max number of traces to use.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    traces_dir = Path(args.traces_dir)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup LM
    import dspy
    from scripts.run_experiments import create_lm
    
    lm_config = {
        "model": args.lm_model,
        "name": args.lm_model,
    }
    api_key = os.environ.get(args.lm_api_key_env)
    if api_key:
        lm_config["api_key"] = api_key
    
    lm = create_lm(lm_config)
    dspy.configure(lm=lm)

    # Build prompt map
    available_prompts = {p.stem for p in prompt_dir.glob("*.md")}
    prompt_map = build_prompt_map(prompt_dir)

    # Parse base prompts
    base_prompts_to_optimize = []
    for raw in args.base_prompts.split(","):
        raw = raw.strip()
        if not raw:
            continue
        key = resolve_prompt_key(raw, prompt_map, available_prompts)
        if key:
            base_prompts_to_optimize.append(key)
        else:
            print(f"[skip] Base prompt '{raw}' has no matching prompt file.")
    
    if not base_prompts_to_optimize:
        print("[error] No valid --base_prompts specified.")
        return

    # Load traces
    all_trace_paths = sorted(traces_dir.glob("*.json"))
    if args.max_traces:
        rng.shuffle(all_trace_paths)
        all_trace_paths = all_trace_paths[:args.max_traces]
        all_trace_paths = sorted(all_trace_paths)
    
    print(f"[info] Loading {len(all_trace_paths)} trace files...")
    all_traces = []
    for trace_path in all_trace_paths:
        trace = collect_full_trace(trace_path)
        if trace:
            all_traces.append(trace)
    print(f"[info] Loaded {len(all_traces)} valid traces")

    # Run TextGrad optimization for each prompt
    optimizer = TextGradOptimizer(lm, prompt_dir, prompt_map, available_prompts)
    
    for prompt_key in base_prompts_to_optimize:
        print(f"\n{'='*60}")
        print(f"Optimizing: {prompt_key}")
        print(f"{'='*60}")
        
        result = optimizer.optimize(all_traces, prompt_key)
        
        if "error" in result:
            print(f"[error] {result['error']}")
            continue
        
        # Save optimized prompt
        out_path = output_dir / f"{prompt_key}.md"
        out_path.write_text(result["new_instructions"].strip() + "\n", encoding="utf-8")
        print(f"[done] Wrote optimized prompt to {out_path}")
        
        # Save gradients for analysis
        gradients_path = output_dir / f"{prompt_key}_gradients.json"
        with open(gradients_path, "w") as f:
            json.dump({
                "base_prompt_key": prompt_key,
                "num_traces": result["num_traces"],
                "gradients": result["gradients"],
            }, f, indent=2)

    # Print cost summary
    total_cost, total_prompt, total_completion, missing_cost = summarize_lm_cost(lm)
    if total_cost is None:
        print("\nLM cost summary: no LM calls recorded.")
    else:
        print(f"\nLM cost summary: "
              f"cost_usd={total_cost:.6f}, "
              f"prompt_tokens={total_prompt}, "
              f"completion_tokens={total_completion}, "
              f"missing_cost_entries={missing_cost}")
    
    print(f"\nWrote optimized prompts to {output_dir}")


if __name__ == "__main__":
    main()
