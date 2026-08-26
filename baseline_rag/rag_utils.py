import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


ERROR_TYPE_RE = re.compile(r"error_type['\"]?\s*[:=]\s*'([^']+)'|\"error_type\"\\s*:\\s*\"([^\"]+)\"")
ERROR_INFO_RE = re.compile(r"error_info['\"]?\s*[:=]\s*'([^']+)'|\"error_info\"\\s*:\\s*\"([^\"]+)\"")
ERROR_TEXT_RE = re.compile(r"Error Text:\\s*(.+)")
ERROR_TYPE_COMMENT_RE = re.compile(r"Error Type:\\s*([A-Za-z0-9_]+)")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\\s*([\\s\\S]*?)```")
JSON_BLOCK_FULL_RE = re.compile(r"```(?:json)?\\s*[\\s\\S]*?```")


def normalize_action_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    if name.endswith(".predict"):
        name = name[:-8]
    return name


def load_prompt_map(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
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
    verusage_dir = Path("verusage")
    if verusage_dir.exists():
        verusage_path = str(verusage_dir.resolve())
        if verusage_path not in sys.path:
            sys.path.insert(0, verusage_path)
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
            matches = re.findall(r"load_prompt\\(\\s*['\"]([^'\"]+)['\"]\\s*\\)", text)
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


def compact_text(text: str, max_chars: int | None) -> str:
    if text is None:
        return ""
    cleaned = re.sub(r"\\s+", " ", text).strip()
    if max_chars and len(cleaned) > max_chars:
        return cleaned[: max_chars - 3] + "..."
    return cleaned


def extract_error_info(text: str) -> dict:
    info = {}
    if not text:
        return info
    match = ERROR_TYPE_RE.search(text)
    if match:
        info["error_type"] = match.group(1) or match.group(2)
    match = ERROR_INFO_RE.search(text)
    if match:
        info["error_info"] = match.group(1) or match.group(2)
    match = ERROR_TEXT_RE.search(text)
    if match:
        info["error_text"] = match.group(1).strip()
    match = ERROR_TYPE_COMMENT_RE.search(text)
    if match and "error_type" not in info:
        info["error_type"] = match.group(1).strip()
    return info


def extract_error_snippet(text: str) -> str:
    if not text:
        return ""
    info_key = "error_info"
    full_key = "full_error"
    start = text.find(info_key)
    end = text.find(full_key)
    if start != -1 and end != -1 and end > start:
        return text[start:end].strip()
    return ""


def extract_json_block(text: str) -> dict | None:
    if not text:
        return None
    if "```" in text:
        blocks = JSON_BLOCK_RE.findall(text)
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


def parse_router_action(text: str) -> str | None:
    data = extract_json_block(text)
    if not isinstance(data, dict):
        return None
    action = data.get("primary_action")
    if not action:
        return None
    return str(action).strip().lower()


def compact_text_keep_json_block(text: str, max_chars: int | None) -> str:
    if text is None:
        return ""
    if not max_chars or len(text) <= max_chars:
        return text
    last_match = None
    for match in JSON_BLOCK_FULL_RE.finditer(text):
        last_match = match
    if last_match:
        json_block = last_match.group(0)
        if len(json_block) >= max_chars:
            return json_block
        prefix_budget = max_chars - len(json_block)
        prefix = text[: last_match.start()]
        if len(prefix) > prefix_budget:
            prefix = prefix[-prefix_budget:]
        return prefix + json_block
    return text[-max_chars:]


def parse_assistant_metadata(text: str) -> dict:
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    choice = None
    for value in data.values():
        if isinstance(value, dict):
            choice = value
            break
    if not choice:
        return {}
    meta = {
        "evaluation_result": choice.get("evaluation result"),
        "accepted": choice.get("accepted"),
        "score": choice.get("score"),
        "added_comments": choice.get("added_comments"),
        "response": choice.get("response"),
    }
    if isinstance(meta["accepted"], str):
        lowered = meta["accepted"].strip().lower()
        if lowered in ("true", "false"):
            meta["accepted"] = lowered == "true"
    extra = extract_error_info(meta.get("added_comments") or "")
    meta.update({k: v for k, v in extra.items() if v})
    return meta


def parse_prompt_descriptions(path: Path) -> dict:
    if not path.exists():
        return {}
    entries = {}
    current = None
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line == "***":
            continue
        if line.endswith(".md"):
            current = line.replace(".md", "")
            entries[current] = {"short": "", "long": ""}
            continue
        if current and not entries[current]["short"]:
            entries[current]["short"] = line
        elif current:
            if entries[current]["long"]:
                entries[current]["long"] += " " + line
            else:
                entries[current]["long"] = line
    return entries


def extract_trace_segments(trace_data: dict) -> list[dict]:
    segments = []
    for key, value in trace_data.items():
        if not key.startswith("sub_agent_calling_"):
            continue
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        trace = value.get("trace", [])
        if not name or not trace:
            continue
        segments.append({"name": name, "trace": trace, "key": key})
    return segments


def build_chunk_text(
    action_name: str,
    prompt_key: str | None,
    trace_file: str,
    trace_key: str,
    trace: list[dict],
    extra_context: dict | None,
    max_system_chars: int | None,
    max_user_chars: int | None,
    max_assistant_chars: int | None,
) -> tuple[str, dict]:
    user_msgs = [m.get("content", "") for m in trace if m.get("role") == "user"]
    assistant_msgs = [m.get("content", "") for m in trace if m.get("role") == "assistant"]

    user_primary_raw = user_msgs[0] if user_msgs else ""
    user_secondary_raw = user_msgs[-1] if len(user_msgs) > 1 else ""
    user_primary = compact_text(user_primary_raw, max_user_chars)
    user_secondary = compact_text(user_secondary_raw, max_user_chars)
    assistant_raw = assistant_msgs[-1] if assistant_msgs else ""
    extra_context = extra_context or {}

    error_info = extract_error_info(user_secondary or user_primary)
    assistant_meta = parse_assistant_metadata(assistant_raw)
    error_info.update({k: v for k, v in assistant_meta.items() if k.startswith("error_")})
    error_snippet = extract_error_snippet(user_secondary_raw or user_primary_raw)
    if action_name == "assertion_reasoning_pipeline":
        assistant_response = assistant_raw
        assistant_text = compact_text_keep_json_block(assistant_response, max_assistant_chars)
    else:
        assistant_response = assistant_meta.get("response") or assistant_raw
        assistant_text = compact_text(assistant_response, max_assistant_chars)

    lines = []
    if error_snippet:
        lines.append("Error Snippet:")
        lines.append(compact_text(error_snippet, max_user_chars))
        if assistant_text:
            lines.append("Assistant Output:")
            lines.append(assistant_text)
    else:
        if extra_context.get("question"):
            lines.append("Question:")
            lines.append(compact_text(str(extra_context["question"]), max_user_chars))
        if extra_context.get("gold_answer"):
            lines.append("Gold Answer:")
            lines.append(compact_text(str(extra_context["gold_answer"]), 400))
        if extra_context.get("prediction_answer"):
            lines.append("Prediction Answer:")
            lines.append(compact_text(str(extra_context["prediction_answer"]), 400))
        if extra_context.get("metric_output") is not None:
            lines.append(f"Metric Output: {extra_context['metric_output']}")
        if assistant_text:
            lines.append("Assistant Output:")
            lines.append(assistant_text)
    if assistant_meta.get("evaluation_result"):
        lines.append(f"Evaluation: {assistant_meta['evaluation_result']}")
    if assistant_meta.get("accepted") is not None:
        lines.append(f"Accepted: {assistant_meta['accepted']}")
    if assistant_meta.get("score"):
        lines.append(f"Score: {assistant_meta['score']}")
    if assistant_meta.get("added_comments"):
        lines.append("Assistant Metadata:")
        lines.append(compact_text(assistant_meta["added_comments"], 400))

    metadata = {
        "error_type": error_info.get("error_type"),
        "error_text": error_info.get("error_text"),
        "error_info": error_info.get("error_info"),
        "evaluation_result": assistant_meta.get("evaluation_result"),
        "accepted": assistant_meta.get("accepted"),
        "score": assistant_meta.get("score"),
        "example_id": extra_context.get("example_id"),
        "metric_output": extra_context.get("metric_output"),
    }
    return "\n".join(lines).strip(), metadata


def load_prompt_text(prompt_dir: Path, prompt_key: str) -> str:
    path = prompt_dir / f"{prompt_key}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def gather_trace_documents(
    traces_dir: Path,
    prompt_dir: Path,
    prompt_map_path: str | None,
    max_system_chars: int | None,
    max_user_chars: int | None,
    max_assistant_chars: int | None,
) -> list[dict]:
    available_prompts = {p.stem for p in prompt_dir.glob("*.md")}
    prompt_map = build_auto_prompt_map(prompt_dir)
    prompt_map.update(load_prompt_map(prompt_map_path))

    documents = []
    doc_id = 0
    for trace_path in sorted(traces_dir.glob("*.json")):
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        extra_context = {
            "example_id": data.get("example_id"),
            "question": data.get("question"),
            "gold_answer": data.get("gold_answer"),
            "metric_output": data.get("metric_output"),
        }
        prediction = data.get("prediction")
        if isinstance(prediction, dict):
            extra_context["prediction_answer"] = prediction.get("answer")
        elif prediction is not None:
            extra_context["prediction_answer"] = prediction
        segments = extract_trace_segments(data)
        action_meta_map = {}
        action_last_assistant = {}
        for segment in segments:
            name = normalize_action_name(segment["name"])
            last_assistant = next(
                (m for m in reversed(segment["trace"]) if m.get("role") == "assistant"), None
            )
            if last_assistant:
                action_last_assistant[name] = last_assistant.get("content", "")
                meta = parse_assistant_metadata(last_assistant.get("content", ""))
                if meta.get("accepted") is not None or meta.get("score") is not None:
                    action_meta_map[name] = meta

        for segment in segments:
            action_name = normalize_action_name(segment["name"])
            prompt_key = resolve_prompt_key(action_name, prompt_map, available_prompts)
            chunk_text, meta = build_chunk_text(
                action_name=action_name,
                prompt_key=prompt_key,
                trace_file=trace_path.name,
                trace_key=segment["key"],
                trace=segment["trace"],
                extra_context=extra_context,
                max_system_chars=max_system_chars,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
            )
            if action_name == "assertion_reasoning_pipeline":
                router_action = parse_router_action(action_last_assistant.get(action_name, ""))
                routed_meta = action_meta_map.get(router_action) if router_action else None
                if routed_meta:
                    for key in ("accepted", "score", "evaluation_result"):
                        if routed_meta.get(key) is not None:
                            meta[key] = routed_meta.get(key)
            documents.append(
                {
                    "doc_id": doc_id,
                    "action_name": action_name,
                    "prompt_key": prompt_key,
                    "trace_file": trace_path.name,
                    "trace_key": segment["key"],
                    "text": chunk_text,
                    **meta,
                }
            )
            doc_id += 1
    return documents


def tokenize_keywords(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)
    stopwords = {
        "error",
        "errors",
        "verus",
        "assert",
        "failed",
        "failure",
        "line",
        "end",
        "body",
        "function",
        "proof",
        "false",
        "true",
    }
    return [t.lower() for t in tokens if len(t) > 2 and t.lower() not in stopwords]


def common_error_keywords(documents: list[dict], top_k: int = 8) -> list[str]:
    counter = Counter()
    for doc in documents:
        for field in ("error_type", "error_text", "evaluation_result"):
            counter.update(tokenize_keywords(doc.get(field) or ""))
    return [tok for tok, _ in counter.most_common(top_k)]


def build_query(
    action_name: str,
    prompt_key: str,
    prompt_text: str,
    prompt_descriptions: dict,
    keywords: list[str],
    max_prompt_chars: int = 800,
) -> str:
    desc = prompt_descriptions.get(prompt_key) or {}
    short = desc.get("short") or ""
    long = desc.get("long") or ""
    parts = [
        f"action {action_name}",
        f"prompt {prompt_key}",
    ]
    if short:
        parts.append(f"role {short}")
    if long:
        parts.append(f"details {long}")
    if keywords:
        parts.append("failures " + ", ".join(keywords))
    if prompt_text:
        parts.append("prompt_excerpt " + compact_text(prompt_text, max_prompt_chars))
    return "\n".join(parts)


def render_evidence(
    documents: list[dict],
    max_doc_chars: int | None,
    max_total_chars: int | None,
) -> tuple[str, list[dict]]:
    blocks = []
    used = []
    total = 0
    for idx, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        if "```json" in text:
            snippet = compact_text_keep_json_block(text, max_doc_chars)
        else:
            snippet = compact_text(text, max_doc_chars)
        header = (
            f"[{idx}] trace={doc.get('trace_file')} {doc.get('trace_key')}; "
            f"error_type={doc.get('error_type')}; accepted={doc.get('accepted')}"
        )
        block = header + "\n" + snippet
        if max_total_chars and total + len(block) > max_total_chars:
            break
        blocks.append(block)
        used.append(doc)
        total += len(block)
    return "\n\n---\n\n".join(blocks), used


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_lm_config(
    path: str | None,
    lm_config_json: str | None,
    lm_model: str | None,
    lm_api_key_env: str | None,
) -> dict:
    if lm_config_json:
        cfg = json.loads(lm_config_json)
    elif path:
        with open(path, "r", encoding="utf-8") as f:
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
