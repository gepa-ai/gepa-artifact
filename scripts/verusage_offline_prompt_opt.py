#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from pathlib import Path

import dspy

from gepa_artifact.gepa.instruction_proposal import ProposeNewInstructionModule


class SimpleSignature(dspy.Signature):
    input: str = dspy.InputField()
    response: str = dspy.OutputField()


def load_lm_config(
    path: str | None,
    lm_config_json: str | None,
    lm_model: str | None,
    lm_api_key_env: str | None,
) -> dict:
    if lm_config_json:
        cfg = json.loads(lm_config_json)
    elif path:
        with open(path, "r") as f:
            cfg = json.load(f)
        cfg = cfg.get("lm_config", cfg)
    else:
        cfg = {}

    if lm_model:
        cfg = dict(cfg)
        cfg["model"] = lm_model
        cfg.setdefault("name", lm_model)
        if "api_key" not in cfg and lm_api_key_env:
            key = os.environ.get(lm_api_key_env)
            if key:
                cfg["api_key"] = key

    if not cfg:
        raise ValueError("Provide --lm_config_path/--lm_config_json or --lm_model.")
    return cfg


def extract_json_block(text: str) -> dict | None:
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", text)
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
        cand = text[start : end + 1]
        try:
            return json.loads(cand)
        except Exception:
            return None
    return None


def parse_action_score(text: str) -> tuple[float, str]:
    data = extract_json_block(text)
    if not isinstance(data, dict):
        return 0.0, "No parsable score JSON found."
    accepted = data.get("accepted")
    score_obj = data.get("score", {})
    eval_result = data.get("evaluation result", "")
    score = 1.0 if accepted else 0.0
    feedback = f"accepted={accepted}; score={score_obj}; evaluation={eval_result}"
    return score, feedback


def parse_router_action(text: str) -> str | None:
    data = extract_json_block(text)
    if not isinstance(data, dict):
        return None
    action = data.get("primary_action")
    if not action:
        return None
    return str(action).strip().lower()


def summarize_lm_cost(lm) -> tuple[float | None, int, int, int]:
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


def normalize_prompt_name(name: str) -> str:
    return name.replace(".predict", "")


def load_prompt_text(prompt_dir: Path, name: str, skip_missing: bool) -> str | None:
    path = prompt_dir / f"{name}.md"
    if not path.exists():
        if skip_missing:
            print(f"[skip] Prompt file not found: {path}")
            return None
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt_map(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--prompt_map_path must be a JSON object.")
    cleaned = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        cleaned[k.replace(".md", "")] = v.replace(".md", "")
    return cleaned


def build_auto_prompt_map(prompt_dir: Path) -> dict:
    prompt_map = {}
    try:
        from verusage.agents.actions.merged_prompt_ablation import MergedPromptAblationAction

        for action_type, prompt_name in MergedPromptAblationAction().action_prompt_mapping.items():
            prompt_map[action_type.value] = prompt_name
    except Exception as exc:
        print(f"[warn] Failed to load merged prompt mapping: {exc}")

    actions_dir = Path("verusage/agents/actions")
    if actions_dir.exists():
        for path in actions_dir.glob("*.py"):
            if path.name.startswith("__"):
                continue
            text = path.read_text(encoding="utf-8")
            matches = re.findall(r'load_prompt\(\s*["\']([^"\']+)["\']\s*\)', text)
            if not matches:
                continue
            action_name = path.stem
            if action_name not in prompt_map:
                prompt_map[action_name] = matches[0]

    prompt_map.setdefault("nonlinear_arithmetic", "arithmetic_reasoning")
    prompt_map.setdefault("invariant_front_repair", "invariant_front_repair_general")
    prompt_map.setdefault("postcondition_repair", "postcondition_repair_basic")
    return prompt_map


def resolve_prompt_key(name: str, prompt_map: dict, available_prompts: set[str]) -> str | None:
    mapped = prompt_map.get(name, name)
    if mapped in available_prompts:
        return mapped
    return None


def collect_segments(trace_path: Path) -> list[dict]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    segments = []
    for key, val in data.items():
        if not key.startswith("sub_agent_calling_"):
            continue
        if not isinstance(val, dict):
            continue
        name = val.get("name")
        trace = val.get("trace", [])
        if not name or not trace:
            continue
        segments.append({"name": name, "trace": trace, "key": key})
    return segments


def select_segments(segments: list[dict], strategy: str) -> list[dict]:
    if strategy == "all":
        return segments
    if strategy == "first":
        return [segments[0]] if segments else []
    if strategy == "last":
        return [segments[-1]] if segments else []
    return [segments[0]] if segments else []


def main():
    parser = argparse.ArgumentParser(description="Offline prompt optimization from VeruSAGE traces.")
    parser.add_argument("--traces_dir", required=True, help="Directory with VeruSAGE trace JSON files.")
    parser.add_argument(
        "--prompt_dir",
        default=str(Path(__file__).resolve().parents[1] / "verusage" / "agents" / "prompts"),
        help="Directory containing VeruSAGE prompt .md files.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write new prompts.")
    parser.add_argument("--lm_config_path", default=None, help="Path to DSPy LM config JSON.")
    parser.add_argument("--lm_config_json", default=None, help="DSPy LM config JSON string.")
    parser.add_argument("--lm_model", default=None, help="Override LM model name (e.g., anthropic/claude-sonnet-4-5-20250929).")
    parser.add_argument(
        "--lm_api_key_env",
        default="ANTHROPIC_API_KEY",
        help="Env var name to read API key from when --lm_model is used.",
    )
    parser.add_argument(
        "--prompt_subset",
        default=None,
        help="Comma-separated prompt names to optimize (e.g., assertion_reasoning_pipeline,instantiate_forall).",
    )
    parser.add_argument(
        "--prompt_map_path",
        default=None,
        help="JSON mapping from action name to prompt file stem (e.g., arithmetic_overflow_repair -> arithmetic_reasoning).",
    )
    parser.add_argument(
        "--skip_missing_prompts",
        action="store_true",
        default=True,
        help="Skip prompts that are missing from prompt_dir (default: true).",
    )
    parser.add_argument(
        "--segment_strategy",
        default="all",
        choices=["all", "first", "last"],
        help="How to select segments if multiple appear per trace.",
    )
    parser.add_argument("--max_samples_per_prompt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    traces_dir = Path(args.traces_dir)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_prompts = {p.stem for p in prompt_dir.glob("*.md")}
    prompt_map = build_auto_prompt_map(prompt_dir)
    prompt_map.update(load_prompt_map(args.prompt_map_path))

    subset = None
    if args.prompt_subset:
        subset = set()
        for raw in args.prompt_subset.split(","):
            raw = raw.strip()
            if not raw:
                continue
            key = resolve_prompt_key(raw, prompt_map, available_prompts)
            if key:
                subset.add(key)
            else:
                print(f"[skip] Prompt subset '{raw}' has no matching prompt file.")

    lm_config = load_lm_config(args.lm_config_path, args.lm_config_json, args.lm_model, args.lm_api_key_env)
    from scripts.run_experiments import create_lm

    instruction_lm = create_lm(lm_config)
    dspy.configure(lm=instruction_lm)

    # Build per-prompt datasets.
    datasets = {}
    for trace_path in sorted(traces_dir.glob("*.json")):
        segments = collect_segments(trace_path)
        if not segments:
            continue

        # Build action scores for router mapping.
        action_scores = {}
        for seg in segments:
            name = normalize_prompt_name(seg["name"])
            prompt_key = resolve_prompt_key(name, prompt_map, available_prompts)
            if not prompt_key:
                continue
            last_assistant = next((m for m in reversed(seg["trace"]) if m.get("role") == "assistant"), None)
            if not last_assistant:
                continue
            score, feedback = parse_action_score(last_assistant.get("content", ""))
            action_scores[prompt_key] = (score, feedback)

        for seg in select_segments(segments, args.segment_strategy):
            name = normalize_prompt_name(seg["name"])
            prompt_key = resolve_prompt_key(name, prompt_map, available_prompts)
            if not prompt_key:
                print(f"[skip] No prompt file for action '{name}' (trace {trace_path.name}).")
                continue
            if subset and prompt_key not in subset:
                continue
            user_msg = next((m for m in reversed(seg["trace"]) if m.get("role") == "user"), None)
            assistant_msg = next((m for m in reversed(seg["trace"]) if m.get("role") == "assistant"), None)
            if not user_msg or not assistant_msg:
                continue

            score = 0.0
            feedback = "No score."
            if name == "assertion_reasoning_pipeline":
                action = parse_router_action(assistant_msg.get("content", ""))
                action_key = (
                    resolve_prompt_key(action, prompt_map, available_prompts) if action else None
                )
                if action_key and action_key in action_scores:
                    score, feedback = action_scores[action_key]
                else:
                    feedback = f"Router selected action '{action}', but no action score found."
            else:
                score, feedback = parse_action_score(assistant_msg.get("content", ""))

            sample = {
                "inputs": {"input": user_msg.get("content", "")},
                "generated_output": {"response": assistant_msg.get("content", "")},
                "feedback": feedback,
                "score": score,
            }
            datasets.setdefault(prompt_key, []).append(sample)

    # Optionally downsample.
    if args.max_samples_per_prompt:
        for name, samples in datasets.items():
            if len(samples) > args.max_samples_per_prompt:
                rng.shuffle(samples)
                datasets[name] = samples[: args.max_samples_per_prompt]

    # Optimize prompts.
    for name, samples in sorted(datasets.items()):
        prompt_text = load_prompt_text(prompt_dir, name, args.skip_missing_prompts)
        if prompt_text is None:
            continue
        base_program = dspy.Predict(SimpleSignature)
        base_program.signature = base_program.signature.with_instructions(prompt_text)
        proposer = ProposeNewInstructionModule(
            base_program=base_program,
            instruction_lm=instruction_lm,
            dataset_with_feedback=samples,
            knowledgebase_qe=None,
        )
        output = proposer.compile()
        new_prompt = output["new_instruction"]
        out_path = output_dir / f"{name}.md"
        out_path.write_text(new_prompt.strip() + "\n", encoding="utf-8")

    total_cost, total_prompt, total_completion, missing_cost = summarize_lm_cost(
        instruction_lm
    )
    if total_cost is None:
        print("LM cost summary: no LM calls recorded.")
    else:
        print(
            "LM cost summary: "
            f"cost_usd={total_cost:.6f}, "
            f"prompt_tokens={total_prompt}, "
            f"completion_tokens={total_completion}, "
            f"missing_cost_entries={missing_cost}"
        )
    print(f"Wrote optimized prompts to {output_dir}")


if __name__ == "__main__":
    main()
