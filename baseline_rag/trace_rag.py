#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import bm25s
import Stemmer

try:
    from baseline_rag.rag_utils import (
        build_auto_prompt_map,
        build_query,
        common_error_keywords,
        gather_trace_documents,
        load_lm_config,
        load_prompt_map,
        load_prompt_text,
        normalize_action_name,
        parse_prompt_descriptions,
        read_jsonl,
        render_evidence,
        resolve_prompt_key,
        write_jsonl,
    )
except ModuleNotFoundError:
    # Support direct execution: python baseline_rag/trace_rag.py
    from rag_utils import (
        build_auto_prompt_map,
        build_query,
        common_error_keywords,
        gather_trace_documents,
        load_lm_config,
        load_prompt_map,
        load_prompt_text,
        normalize_action_name,
        parse_prompt_descriptions,
        read_jsonl,
        render_evidence,
        resolve_prompt_key,
        write_jsonl,
    )


def _save_bm25_index(
    corpus: list[str],
    output_dir: Path,
    k1: float,
    b: float,
    stopwords: str,
    stemmer_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stemmer = Stemmer.Stemmer(stemmer_name)
    tokens = bm25s.tokenize(corpus, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(tokens)
    retriever.save(str(output_dir))
    config = {
        "k1": k1,
        "b": b,
        "stopwords": stopwords,
        "stemmer": stemmer_name,
    }
    (output_dir / "bm25_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def _load_bm25_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"k1": 0.9, "b": 0.4, "stopwords": "en", "stemmer": "english"}


def _retrieve_top_k(
    documents: list[dict],
    query: str,
    top_k: int,
    index_dir: Path | None,
    stopwords: str,
    stemmer_name: str,
    k1: float,
    b: float,
) -> list[dict]:
    if not documents:
        return []
    k = min(top_k, len(documents))
    corpus = [doc["text"] for doc in documents]
    if index_dir and (index_dir / "bm25s_retriever").exists():
        retriever = bm25s.BM25.load(str(index_dir / "bm25s_retriever"))
        config = _load_bm25_config(index_dir / "bm25_config.json")
        stopwords = config.get("stopwords", stopwords)
        stemmer_name = config.get("stemmer", stemmer_name)
    else:
        stemmer = Stemmer.Stemmer(stemmer_name)
        tokens = bm25s.tokenize(corpus, stopwords=stopwords, stemmer=stemmer, show_progress=False)
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(tokens)
    stemmer = Stemmer.Stemmer(stemmer_name)
    query_tokens = bm25s.tokenize(query, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=k, n_threads=1, show_progress=False)
    if results is None or len(results) == 0:
        return []
    ranked = []
    for idx in results[0]:
        if 0 <= idx < len(documents):
            ranked.append(documents[idx])
    return ranked


def _compose_static_prompt(base_prompt: str, evidence: str) -> str:
    header = "## Offline Trace Evidence (retrieved, same action only)"
    guidance = (
        "Use the evidence above to guide repair tactics and formatting. "
        "Do not quote the evidence verbatim in your response."
    )
    return base_prompt.strip() + "\n\n" + header + "\n" + evidence + "\n\n" + guidance + "\n"


def _generate_with_teacher_lm(base_prompt: str, evidence: str, lm_config: dict) -> str:
    from scripts.run_experiments import create_lm
    import dspy

    system_prompt = (
        "You are a prompt editor for Verus verification repair actions. "
        "Revise the base prompt using the evidence. Preserve the style and "
        "constraints, and add any missing rules or steps. "
        "Return the full revised prompt only."
    )
    user_content = (
        "Base prompt:\n"
        f"{base_prompt.strip()}\n\n"
        "Evidence:\n"
        f"{evidence.strip()}\n\n"
        "Output: revised prompt only."
    )
    lm = create_lm(lm_config)
    dspy.configure(lm=lm)
    outputs = lm(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])
    if not outputs:
        raise RuntimeError("Teacher LM returned no outputs.")
    return outputs[0].strip()


def build_index(args: argparse.Namespace) -> None:
    traces_dir = Path(args.traces_dir)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = gather_trace_documents(
        traces_dir=traces_dir,
        prompt_dir=prompt_dir,
        prompt_map_path=args.prompt_map_path,
        max_system_chars=args.max_system_chars,
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
    )
    if args.filter_accepted:
        documents = [doc for doc in documents if doc.get("accepted") is True]

    write_jsonl(output_dir / "documents.jsonl", documents)
    meta = {
        "num_documents": len(documents),
        "num_traces": len(list(traces_dir.glob("*.json"))),
    }
    (output_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.build_global_index:
        _save_bm25_index(
            corpus=[doc["text"] for doc in documents],
            output_dir=output_dir / "bm25s_retriever",
            k1=args.k1,
            b=args.b,
            stopwords=args.stopwords,
            stemmer_name=args.stemmer,
        )

    actions_dir = output_dir / "actions"
    for prompt_key in sorted({doc["prompt_key"] for doc in documents if doc.get("prompt_key")}):
        action_docs = [doc for doc in documents if doc.get("prompt_key") == prompt_key]
        action_dir = actions_dir / prompt_key
        write_jsonl(action_dir / "documents.jsonl", action_docs)
        _save_bm25_index(
            corpus=[doc["text"] for doc in action_docs],
            output_dir=action_dir / "bm25s_retriever",
            k1=args.k1,
            b=args.b,
            stopwords=args.stopwords,
            stemmer_name=args.stemmer,
        )

    print(f"Wrote {len(documents)} documents to {output_dir}")


def build_prompt(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    prompt_dir = Path(args.prompt_dir)
    prompt_descriptions = parse_prompt_descriptions(Path(args.prompt_descriptions_path))

    available_prompts = {p.stem for p in prompt_dir.glob("*.md")}
    prompt_map = build_auto_prompt_map(prompt_dir)
    prompt_map.update(load_prompt_map(args.prompt_map_path))

    action_name = normalize_action_name(args.action)
    prompt_key = resolve_prompt_key(action_name, prompt_map, available_prompts) or action_name
    if prompt_key not in available_prompts:
        raise FileNotFoundError(f"Prompt file not found for action '{args.action}'.")

    base_prompt = load_prompt_text(prompt_dir, prompt_key)
    action_docs_path = index_dir / "actions" / prompt_key / "documents.jsonl"
    if action_docs_path.exists():
        documents = read_jsonl(action_docs_path)
        action_index_dir = index_dir / "actions" / prompt_key
    else:
        documents = [d for d in read_jsonl(index_dir / "documents.jsonl") if d.get("prompt_key") == prompt_key]
        action_index_dir = None
    if args.filter_accepted:
        documents = [doc for doc in documents if doc.get("accepted") is True]

    if not documents:
        raise ValueError(f"No trace documents found for prompt '{prompt_key}'.")

    keywords = common_error_keywords(documents, top_k=args.keyword_top_k)
    query = build_query(
        action_name=action_name,
        prompt_key=prompt_key,
        prompt_text=base_prompt,
        prompt_descriptions=prompt_descriptions,
        keywords=keywords,
        max_prompt_chars=args.max_prompt_chars,
    )

    ranked_docs = _retrieve_top_k(
        documents=documents,
        query=query,
        top_k=args.top_k,
        index_dir=action_index_dir,
        stopwords=args.stopwords,
        stemmer_name=args.stemmer,
        k1=args.k1,
        b=args.b,
    )

    evidence, used_docs = render_evidence(
        documents=ranked_docs,
        max_doc_chars=args.max_doc_chars,
        max_total_chars=args.max_total_chars,
    )

    if args.evidence_path:
        Path(args.evidence_path).write_text(evidence + "\n", encoding="utf-8")

    if args.lm_config_path or args.lm_config_json or args.lm_model:
        lm_config = load_lm_config(
            args.lm_config_path, args.lm_config_json, args.lm_model, args.lm_api_key_env
        )
        prompt_text = _generate_with_teacher_lm(base_prompt, evidence, lm_config)
    else:
        prompt_text = _compose_static_prompt(base_prompt, evidence)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt_text.strip() + "\n", encoding="utf-8")

    meta = {
        "prompt_key": prompt_key,
        "action_name": action_name,
        "top_k": args.top_k,
        "num_used_docs": len(used_docs),
        "keyword_top_k": args.keyword_top_k,
    }
    (output_path.parent / "prompt_rag_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote new prompt to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline trace-RAG baseline for VeruSAGE prompts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build per-action BM25 index from traces.")
    build_parser.add_argument("--traces_dir", required=True, help="Directory with trace JSON files.")
    build_parser.add_argument(
        "--prompt_dir",
        default="verusage/agents/prompts",
        help="Directory containing action prompt .md files.",
    )
    build_parser.add_argument("--prompt_map_path", default=None, help="Optional JSON action->prompt map.")
    build_parser.add_argument("--output_dir", required=True, help="Directory to write index data.")
    build_parser.add_argument("--max_system_chars", type=int, default=800)
    build_parser.add_argument("--max_user_chars", type=int, default=1600)
    build_parser.add_argument("--max_assistant_chars", type=int, default=1600)
    build_parser.add_argument(
        "--filter_accepted",
        action="store_true",
        default=False,
        help="Keep only trace chunks with accepted=true (useful for verusage).",
    )
    build_parser.add_argument("--stopwords", default="en")
    build_parser.add_argument("--stemmer", default="english")
    build_parser.add_argument("--k1", type=float, default=0.9)
    build_parser.add_argument("--b", type=float, default=0.4)
    build_parser.add_argument("--build_global_index", action="store_true", default=False)
    build_parser.set_defaults(func=build_index)

    prompt_parser = subparsers.add_parser("build-prompt", help="Build a static prompt using offline retrieval.")
    prompt_parser.add_argument("--index_dir", required=True, help="Index directory created by build-index.")
    prompt_parser.add_argument("--action", required=True, help="Action name or prompt file stem.")
    prompt_parser.add_argument(
        "--prompt_dir",
        default="verusage/agents/prompts",
        help="Directory containing action prompt .md files.",
    )
    prompt_parser.add_argument(
        "--prompt_descriptions_path",
        default="verusage/agents/prompt_descriptions.txt",
        help="Path to prompt_descriptions.txt.",
    )
    prompt_parser.add_argument("--prompt_map_path", default=None, help="Optional JSON action->prompt map.")
    prompt_parser.add_argument("--output_path", required=True, help="Path to write the new prompt.")
    prompt_parser.add_argument("--top_k", type=int, default=3)
    prompt_parser.add_argument("--keyword_top_k", type=int, default=8)
    prompt_parser.add_argument("--max_doc_chars", type=int, default=1200)
    prompt_parser.add_argument("--max_total_chars", type=int, default=6000)
    prompt_parser.add_argument("--max_prompt_chars", type=int, default=800)
    prompt_parser.add_argument(
        "--filter_accepted",
        action="store_true",
        default=False,
        help="Filter to accepted=true chunks before retrieval.",
    )
    prompt_parser.add_argument("--stopwords", default="en")
    prompt_parser.add_argument("--stemmer", default="english")
    prompt_parser.add_argument("--k1", type=float, default=0.9)
    prompt_parser.add_argument("--b", type=float, default=0.4)
    prompt_parser.add_argument("--evidence_path", default=None)

    prompt_parser.add_argument("--lm_config_path", default=None)
    prompt_parser.add_argument("--lm_config_json", default=None)
    prompt_parser.add_argument("--lm_model", default=None)
    prompt_parser.add_argument("--lm_api_key_env", default="ANTHROPIC_API_KEY")
    prompt_parser.set_defaults(func=build_prompt)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
