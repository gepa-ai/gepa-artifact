#!/usr/bin/env python3
"""Build comparable WebArena prompt baselines from offline traces."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_PROMPT = Path("webarena/p_cot_id_actree_2s.json")
DEFAULT_SITES = ("gitlab", "map", "reddit", "shopping", "shopping_admin")
DEFAULT_BASELINES = ("naive_fewshot", "rag", "textgrad")
REQUIRED_PROMPT_KEYS = ("intro", "examples", "template", "meta_data")
EVIDENCE_HEADER = "Offline Site-Specific Trace Evidence"


TEXTGRAD_TRACE_PROMPT = """You are analyzing a failed WebArena browser-agent trace to improve only the agent's intro instructions.

Current intro instructions:
<current_intro>
{current_intro}
</current_intro>

Site: {site}
Task ID: {task_id}
Objective: {objective}
Score: {score}

Failed trace:
<trace>
{trace_text}
</trace>

Identify concrete prompt-level issues that contributed to the failure. Focus on missing site-specific guidance, bad reasoning patterns, navigation mistakes, extraction mistakes, and stopping mistakes. Do not suggest changing the examples, template, metadata, or action grammar.

Return only a concise bullet list of textual gradients.
"""


TEXTGRAD_APPLY_PROMPT = """I provided a WebArena browser agent with these intro instructions:

<current_intro>
{current_intro}
</current_intro>

I ran the agent on {num_traces} failed same-site traces and collected textual gradients:

<gradients>
{gradients}
</gradients>

Rewrite only the intro instructions to incorporate the gradients.

Requirements:
1. Keep the result as a complete replacement for the intro text, not a diff.
2. Preserve the existing action syntax and the stop-action convention.
3. Do not mention that the instructions came from traces, gradients, or offline optimization.
4. Do not add examples or template placeholders.
5. Keep the instructions general enough to work across tasks on this site.

Output the complete rewritten intro between <BEGIN> and <END> tags.
"""


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def require_prompt_shape(prompt: dict[str, Any], path: Path) -> None:
    missing = [key for key in REQUIRED_PROMPT_KEYS if key not in prompt]
    if missing:
        raise ValueError(f"Prompt {path} is missing required keys: {missing}")
    if not isinstance(prompt["intro"], str):
        raise ValueError(f"Prompt {path} has non-string intro")
    if not isinstance(prompt["examples"], list):
        raise ValueError(f"Prompt {path} has non-list examples")
    if not isinstance(prompt["template"], str):
        raise ValueError(f"Prompt {path} has non-string template")
    if not isinstance(prompt["meta_data"], dict):
        raise ValueError(f"Prompt {path} has non-dict meta_data")


def load_base_prompt(path: Path) -> dict[str, Any]:
    prompt = load_json(path)
    if not isinstance(prompt, dict):
        raise ValueError(f"Base prompt must be a JSON object: {path}")
    require_prompt_shape(prompt, path)
    return prompt


def load_site_traces(site_dir: Path) -> list[dict[str, Any]]:
    trace_paths = sorted(site_dir.joinpath("traces").glob("trace_*.json"))
    if not trace_paths:
        raise ValueError(f"No trace_*.json files found under {site_dir / 'traces'}")

    traces: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        trace = load_json(trace_path)
        if not isinstance(trace, dict):
            raise ValueError(f"Trace must be a JSON object: {trace_path}")
        trace = dict(trace)
        trace["_trace_path"] = str(trace_path)
        traces.append(trace)
    return traces


def trace_score(trace: dict[str, Any]) -> float:
    try:
        return float(trace.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_success(trace: dict[str, Any]) -> bool:
    return trace_score(trace) >= 1.0


def is_stop_step(step: dict[str, Any]) -> bool:
    action_type = str(step.get("action_type", "")).upper()
    action_str = str(step.get("action_str", "")).strip().lower()
    return "STOP" in action_type or action_str.startswith("stop")


def get_previous_action(trace: dict[str, Any], step: dict[str, Any], fallback_index: int | None = None) -> str:
    action_history = trace.get("action_history") or []
    if not isinstance(action_history, list):
        action_history = []

    step_number = step.get("step")
    try:
        history_index = int(step_number) - 1
    except (TypeError, ValueError):
        history_index = fallback_index if fallback_index is not None else 0

    if 0 <= history_index < len(action_history):
        return str(action_history[history_index])
    return "None"


def select_last_step(trace: dict[str, Any], *, require_non_stop: bool) -> tuple[dict[str, Any], int] | None:
    steps = trace.get("steps") or []
    if not isinstance(steps, list):
        return None
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        if not isinstance(step, dict):
            continue
        if require_non_stop and is_stop_step(step):
            continue
        return step, index
    return None


def format_step_input(template: str, trace: dict[str, Any], step: dict[str, Any], step_index: int) -> str:
    return template.format(
        observation=step.get("observation", ""),
        url=step.get("url", ""),
        objective=trace.get("intent", ""),
        previous_action=get_previous_action(trace, step, step_index),
    )


def collect_fewshot_candidates(
    traces: list[dict[str, Any]], template: str, *, require_success: bool
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for trace in traces:
        if is_success(trace) != require_success:
            continue
        selected = select_last_step(trace, require_non_stop=True)
        if selected is None:
            continue
        step, step_index = selected
        raw_prediction = str(step.get("raw_prediction", "")).strip()
        if not raw_prediction:
            continue
        formatted_input = format_step_input(template, trace, step, step_index)
        candidates.append((formatted_input, raw_prediction, trace))
    return candidates


def build_naive_fewshot_prompt(
    base_prompt: dict[str, Any], traces: list[dict[str, Any]], *, max_examples: int, seed: int
) -> dict[str, Any]:
    prompt = copy.deepcopy(base_prompt)
    rng = random.Random(seed)

    successes = collect_fewshot_candidates(traces, prompt["template"], require_success=True)
    failures = collect_fewshot_candidates(traces, prompt["template"], require_success=False)
    rng.shuffle(successes)
    rng.shuffle(failures)

    selected = successes[:max_examples]
    if len(selected) < max_examples:
        selected.extend(failures[: max_examples - len(selected)])
    if len(selected) < max_examples:
        raise ValueError(f"Only found {len(selected)} usable few-shot examples; need {max_examples}")

    prompt["examples"] = list(prompt["examples"]) + [[inp, out] for inp, out, _trace in selected]
    return prompt


def clip_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 18].rstrip() + "\n[...truncated...]"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    tokenized_docs = [tokenize(doc) for doc in documents]
    query_terms = tokenize(query)
    if not documents or not query_terms:
        return [0.0 for _ in documents]

    doc_count = len(tokenized_docs)
    doc_freq: dict[str, int] = {}
    for doc in tokenized_docs:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for doc in tokenized_docs:
        doc_len = len(doc) or 1
        term_counts: dict[str, int] = {}
        for term in doc:
            term_counts[term] = term_counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1.0 - b + b * doc_len / max(avg_doc_len, 1.0))
            score += idf * freq * (k1 + 1.0) / denom
        scores.append(score)
    return scores


def trace_label(trace: dict[str, Any]) -> str:
    return "success" if is_success(trace) else "failure"


def build_evidence_item(trace: dict[str, Any]) -> dict[str, Any] | None:
    selected = select_last_step(trace, require_non_stop=False)
    if selected is None:
        return None
    step, step_index = selected
    action_history = trace.get("action_history") or []
    if not isinstance(action_history, list):
        action_history = []
    recent_actions = " | ".join(str(action) for action in action_history[-4:])
    text = "\n".join(
        [
            f"Outcome: {trace_label(trace)} score={trace_score(trace):g}",
            f"Task ID: {trace.get('task_id', '')}",
            f"Objective: {trace.get('intent', '')}",
            f"URL: {step.get('url', '')}",
            f"Previous action: {get_previous_action(trace, step, step_index)}",
            f"Chosen action: {step.get('action_str', '')}",
            f"Recent actions: {recent_actions}",
            f"Prediction rationale: {clip_text(step.get('raw_prediction', ''), 900)}",
            f"Observation excerpt: {clip_text(step.get('observation', ''), 900)}",
        ]
    )
    return {"trace": trace, "text": text, "outcome": trace_label(trace)}


def select_rag_evidence(traces: list[dict[str, Any]], *, site: str, top_k: int) -> list[dict[str, Any]]:
    items = [item for trace in traces if (item := build_evidence_item(trace)) is not None]
    if not items:
        raise ValueError(f"No usable evidence snippets found for {site}")

    documents = [item["text"] for item in items]
    query = " ".join([site.replace("_", " ")] + [str(trace.get("intent", "")) for trace in traces])
    scores = bm25_scores(query, documents)
    ranked = sorted(range(len(items)), key=lambda idx: (-scores[idx], str(items[idx]["trace"].get("_trace_path", ""))))
    selected_indices = ranked[:top_k]

    if top_k >= 2:
        selected_outcomes = {items[idx]["outcome"] for idx in selected_indices}
        for outcome in ("success", "failure"):
            if outcome in selected_outcomes:
                continue
            replacement = next((idx for idx in ranked if items[idx]["outcome"] == outcome), None)
            if replacement is None or replacement in selected_indices:
                continue
            selected_indices = selected_indices[:-1] + [replacement]
            selected_outcomes.add(outcome)

    selected_indices = sorted(selected_indices, key=lambda idx: (-scores[idx], str(items[idx]["trace"].get("_trace_path", ""))))
    for idx in selected_indices:
        items[idx]["score"] = scores[idx]
    return [items[idx] for idx in selected_indices]


def build_rag_section(site: str, evidence_items: list[dict[str, Any]]) -> str:
    lines = [
        EVIDENCE_HEADER,
        "Use these offline same-site trace snippets as static guidance when they are relevant to the current objective. They are not current page state and should not override the live observation.",
        "",
    ]
    for idx, item in enumerate(evidence_items, 1):
        trace = item["trace"]
        lines.append(
            f"Evidence {idx} ({site}, {item['outcome']}, task_id={trace.get('task_id', '')}, bm25={item.get('score', 0.0):.3f}):"
        )
        lines.append(item["text"])
        lines.append("")
    return "\n".join(lines).rstrip()


def build_rag_prompt(
    base_prompt: dict[str, Any], traces: list[dict[str, Any]], *, site: str, top_k: int
) -> dict[str, Any]:
    prompt = copy.deepcopy(base_prompt)
    evidence_items = select_rag_evidence(traces, site=site, top_k=top_k)
    prompt["intro"] = prompt["intro"].rstrip() + "\n\n" + build_rag_section(site, evidence_items)
    return prompt


def format_trace_for_textgrad(trace: dict[str, Any]) -> str:
    lines = [
        f"Trace path: {trace.get('_trace_path', '')}",
        f"Action history: {' | '.join(str(action) for action in (trace.get('action_history') or []))}",
        "",
    ]
    steps = trace.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        lines.extend(
            [
                f"Step {step.get('step', index)}",
                f"URL: {step.get('url', '')}",
                f"Previous action: {get_previous_action(trace, step, index - 1)}",
                f"Chosen action: {step.get('action_str', '')}",
                f"Raw prediction: {clip_text(step.get('raw_prediction', ''), 1200)}",
                f"Observation: {clip_text(step.get('observation', ''), 1600)}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def select_failed_traces(traces: list[dict[str, Any]], *, max_traces: int, seed: int) -> list[dict[str, Any]]:
    failed = [trace for trace in traces if not is_success(trace)]
    if not failed:
        raise ValueError("No failed traces available for TextGrad")
    rng = random.Random(seed)
    rng.shuffle(failed)
    return failed[:max_traces]


def extract_between_tags(text: str, start_tag: str = "<BEGIN>", end_tag: str = "<END>") -> str | None:
    pattern = re.escape(start_tag) + r"([\s\S]*?)" + re.escape(end_tag)
    match = re.search(pattern, text or "")
    if match:
        return match.group(1).strip()
    return None


def load_lm_config(lm_config_json: str | None) -> dict[str, Any]:
    if lm_config_json:
        cfg = json.loads(lm_config_json)
        if not isinstance(cfg, dict):
            raise ValueError("--lm_config_json must decode to a JSON object")
        return resolve_env_references(cfg)

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.experiment_configs import LM_CONFIGS  # type: ignore

    if not LM_CONFIGS:
        raise ValueError("scripts.experiment_configs.LM_CONFIGS is empty; pass --lm_config_json")
    return resolve_env_references(copy.deepcopy(LM_CONFIGS[0]))


def resolve_env_references(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: resolve_env_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_references(item) for item in value]
    if isinstance(value, str) and value.startswith("env:"):
        env_name = value[4:]
        env_value = os.environ.get(env_name)
        if env_value is None:
            raise ValueError(f"LM config references unset environment variable: {env_name}")
        return env_value
    return value


def create_lm(lm_config: dict[str, Any]) -> Any:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.run_experiments import create_lm as create_repo_lm  # type: ignore

    return create_repo_lm(lm_config)


def call_lm(lm: Any, prompt: str, *, max_tokens: int) -> str:
    try:
        response = lm(prompt, max_tokens=max_tokens, temperature=0.0)
    except TypeError:
        response = lm(prompt, max_tokens=max_tokens)
    if isinstance(response, list):
        return str(response[0] if response else "")
    return str(response)


def build_textgrad_prompt(
    base_prompt: dict[str, Any],
    traces: list[dict[str, Any]],
    *,
    site: str,
    max_traces: int,
    seed: int,
    lm: Any,
) -> dict[str, Any]:
    selected_traces = select_failed_traces(traces, max_traces=max_traces, seed=seed)
    gradients: list[str] = []
    for idx, trace in enumerate(selected_traces, 1):
        prompt = TEXTGRAD_TRACE_PROMPT.format(
            current_intro=base_prompt["intro"],
            site=site,
            task_id=trace.get("task_id", ""),
            objective=trace.get("intent", ""),
            score=trace_score(trace),
            trace_text=format_trace_for_textgrad(trace),
        )
        print(f"[textgrad] {site}: gradient {idx}/{len(selected_traces)} task_id={trace.get('task_id', '')}")
        gradient = call_lm(lm, prompt, max_tokens=1200).strip()
        gradients.append(gradient)

    gradient_block = "\n\n".join(
        f"=== Trace {idx}: task_id={trace.get('task_id', '')} ===\n{gradient}"
        for idx, (trace, gradient) in enumerate(zip(selected_traces, gradients, strict=True), 1)
    )
    apply_prompt = TEXTGRAD_APPLY_PROMPT.format(
        current_intro=base_prompt["intro"],
        num_traces=len(selected_traces),
        gradients=gradient_block,
    )
    print(f"[textgrad] {site}: applying {len(gradients)} gradients")
    rewritten = call_lm(lm, apply_prompt, max_tokens=4096).strip()
    new_intro = extract_between_tags(rewritten) or rewritten
    if not new_intro:
        raise ValueError(f"TextGrad returned an empty intro for {site}")

    prompt_config = copy.deepcopy(base_prompt)
    prompt_config["intro"] = new_intro
    return prompt_config


def output_path_for(base_prompt_path: Path, site_dir: Path, baseline: str) -> Path:
    return site_dir / f"{base_prompt_path.stem}_{baseline}.json"


def build_site_baseline(
    baseline: str,
    base_prompt: dict[str, Any],
    base_prompt_path: Path,
    site: str,
    site_dir: Path,
    traces: list[dict[str, Any]],
    args: argparse.Namespace,
    lm: Any | None,
) -> Path:
    out_path = output_path_for(base_prompt_path, site_dir, baseline)

    if args.dry_run:
        if baseline == "naive_fewshot":
            successes = collect_fewshot_candidates(traces, base_prompt["template"], require_success=True)
            print(f"[dry-run] {site}/{baseline}: would append {args.max_fewshot_examples} examples from {len(successes)} successful candidates -> {out_path}")
        elif baseline == "rag":
            evidence = select_rag_evidence(traces, site=site, top_k=args.rag_top_k)
            print(f"[dry-run] {site}/{baseline}: would append {len(evidence)} BM25 evidence snippets -> {out_path}")
        elif baseline == "textgrad":
            failed = [trace for trace in traces if not is_success(trace)]
            print(f"[dry-run] {site}/{baseline}: would call LM on {min(len(failed), args.textgrad_max_traces)} failed traces -> {out_path}")
        return out_path

    if baseline == "naive_fewshot":
        prompt_config = build_naive_fewshot_prompt(
            base_prompt,
            traces,
            max_examples=args.max_fewshot_examples,
            seed=args.seed,
        )
    elif baseline == "rag":
        prompt_config = build_rag_prompt(base_prompt, traces, site=site, top_k=args.rag_top_k)
    elif baseline == "textgrad":
        if lm is None:
            raise ValueError("TextGrad requires an LM")
        prompt_config = build_textgrad_prompt(
            base_prompt,
            traces,
            site=site,
            max_traces=args.textgrad_max_traces,
            seed=args.seed,
            lm=lm,
        )
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    require_prompt_shape(prompt_config, out_path)
    write_json(out_path, prompt_config)
    print(f"[write] {out_path}")
    return out_path


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[str], Path]:
    sites = parse_csv(args.sites)
    baselines = parse_csv(args.baselines)
    unknown_sites = [site for site in sites if site not in DEFAULT_SITES]
    if unknown_sites:
        raise ValueError(f"Unsupported WebArena site(s) for baseline generation: {unknown_sites}")
    unknown_baselines = [baseline for baseline in baselines if baseline not in DEFAULT_BASELINES]
    if unknown_baselines:
        raise ValueError(f"Unsupported baseline(s): {unknown_baselines}")
    base_prompt_path = Path(args.base_prompt)
    if not base_prompt_path.is_absolute():
        base_prompt_path = PROJECT_ROOT / base_prompt_path
    return sites, baselines, base_prompt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WebArena baseline prompt JSON files from offline traces.")
    parser.add_argument("--sites", default=",".join(DEFAULT_SITES), help="Comma-separated sites to process.")
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES), help="Comma-separated baselines to build.")
    parser.add_argument("--base_prompt", default=str(DEFAULT_BASE_PROMPT), help="Shared base prompt JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--max_fewshot_examples", type=int, default=3, help="Naive few-shot examples per site.")
    parser.add_argument("--rag_top_k", type=int, default=8, help="Number of static RAG evidence snippets per site.")
    parser.add_argument("--textgrad_max_traces", type=int, default=10, help="Maximum failed traces per site for TextGrad.")
    parser.add_argument("--lm_config_json", default=None, help="Optional LM config JSON for TextGrad.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned work without writing files or calling the LM.")
    args = parser.parse_args()

    sites, baselines, base_prompt_path = validate_args(args)
    base_prompt = load_base_prompt(base_prompt_path)

    lm = None
    if "textgrad" in baselines and not args.dry_run:
        lm_config = load_lm_config(args.lm_config_json)
        redacted = dict(lm_config)
        for key in ("api_key", "openai_api_key", "anthropic_api_key"):
            if redacted.get(key):
                redacted[key] = "<redacted>"
        print(f"[textgrad] using LM config: {json.dumps(redacted, sort_keys=True)}")
        lm = create_lm(lm_config)
        import dspy  # type: ignore

        dspy.configure(lm=lm)

    written: list[Path] = []
    for site in sites:
        site_dir = PROJECT_ROOT / "webarena" / site
        traces = load_site_traces(site_dir)
        success_count = sum(1 for trace in traces if is_success(trace))
        print(f"[site] {site}: {len(traces)} traces ({success_count} success, {len(traces) - success_count} failed)")
        for baseline in baselines:
            written.append(build_site_baseline(baseline, base_prompt, base_prompt_path, site, site_dir, traces, args, lm))

    if args.dry_run:
        print(f"[dry-run] planned {len(written)} output file(s)")
    else:
        print(f"[done] wrote {len(written)} output file(s)")


if __name__ == "__main__":
    main()
