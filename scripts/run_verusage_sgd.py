#!/usr/bin/env python3
"""
VeruSAGE GEPA-SGD Prompt Optimization

用法:
    python3 scripts/run_verusage_sgd.py \
        --traces_dir verusage/train_5 \
        --prompt_names postcondition_repair_basic,assertion_reasoning_pipeline \
        --minibatch_size 3 \
        --output_dir verusage/prompts_sgd_train5

输出:
    - 优化后的 prompts（.md 文件）
    - cost_summary.json（Claude 花费统计）
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import dspy
from gepa_artifact.gepa.instruction_proposal import ProposeNewInstructionModule


class SimpleSignature(dspy.Signature):
    input: str = dspy.InputField()
    response: str = dspy.OutputField()


def count_tokens(text: str) -> int:
    """计算 token 数"""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def estimate_sample_tokens(sample: Dict) -> int:
    """估算单个样本的 token 数"""
    total = 0
    # inputs
    for v in sample.get("inputs", {}).values():
        total += count_tokens(str(v))
    # outputs
    for v in sample.get("generated_output", {}).values():
        total += count_tokens(str(v))
    # feedback
    total += count_tokens(sample.get("feedback", ""))
    # 加上格式化开销（markdown headers 等）
    total += 50
    return total


def extract_json_block(text: str) -> Optional[dict]:
    """从文本中提取 JSON 块"""
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        for blk in blocks:
            blk = blk.strip()
            if blk.startswith("{") and blk.endswith("}"):
                try:
                    return json.loads(blk)
                except json.JSONDecodeError:
                    pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            return None
    return None


def parse_action_score(text: str) -> Tuple[float, str]:
    """解析 action 的分数和反馈"""
    data = extract_json_block(text)
    if not isinstance(data, dict):
        return 0.0, "No parsable score JSON found."
    accepted = data.get("accepted")
    score_obj = data.get("score", {})
    eval_result = data.get("evaluation result", "")
    score = 1.0 if accepted else 0.0
    feedback = f"accepted={accepted}; score={score_obj}; evaluation={eval_result}"
    return score, feedback


def collect_segments_from_trace(trace_path: Path) -> List[Dict]:
    """从 trace 文件中收集所有 segment"""
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
        segments.append({"name": name, "trace": trace, "key": key, "source_file": trace_path.name})
    return segments


def build_sample_from_segment(seg: Dict) -> Optional[Dict]:
    """从 segment 构建训练样本"""
    user_msg = next((m for m in reversed(seg["trace"]) if m.get("role") == "user"), None)
    assistant_msg = next((m for m in reversed(seg["trace"]) if m.get("role") == "assistant"), None)
    
    if not user_msg or not assistant_msg:
        return None
    
    score, feedback = parse_action_score(assistant_msg.get("content", ""))
    
    return {
        "inputs": {"input": user_msg.get("content", "")},
        "generated_output": {"response": assistant_msg.get("content", "")},
        "feedback": feedback,
        "score": score,
        "action_name": seg["name"],
        "source_file": seg.get("source_file", ""),
    }


def load_samples_for_action(traces_dir: Path, action_name: str) -> List[Dict]:
    """从目录加载指定 action 类型的所有样本"""
    all_samples = []
    for trace_path in sorted(traces_dir.glob("*.json")):
        segments = collect_segments_from_trace(trace_path)
        for seg in segments:
            if seg["name"] == action_name:
                sample = build_sample_from_segment(seg)
                if sample:
                    all_samples.append(sample)
    return all_samples


def setup_lm():
    """配置 Claude Sonnet 4.5"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("请设置 ANTHROPIC_API_KEY 环境变量")
    
    lm = dspy.LM(
        model="anthropic/claude-sonnet-4-5-20250929",
        api_key=api_key,
        temperature=1,
        max_tokens=16384  # Claude Sonnet 4.5 最大输出
    )
    return lm


# Action 名称到 Prompt 文件名的映射
ACTION_TO_PROMPT = {
    "postcondition_repair": "postcondition_repair_basic",
    "assertion_reasoning_pipeline": "assertion_reasoning_pipeline",
    "uselemma": "uselemma",
    "instantiate_exists": "instantiate_exists",
    "instantiate_forall": "instantiate_forall",
    "case_analysis": "case_analysis",
    "seqsetmap": "seqsetmap",
    "compute": "compute",
    "extensional_equality": "extensional_equality",
    "reveal_opaque": "reveal_opaque",
    "inductive_lemma": "inductive_lemma",
    "induction": "induction",
    "loopinv": "loopinv",
    "nonlinear_arithmetic": "arithmetic_reasoning",
    "decfailend_repair": "postcondition_repair_basic",  # 使用相同的 prompt
    "add_trigger_assert": "add_trigger_assert",
    "precondition_repair": "precondition_repair",
    "type_repair": "type_repair",
}

# Prompt 名称到 Action 名称的映射（当多个 prompt 对应同一个 action 时）
PROMPT_TO_ACTIONS = {
    "postcondition_repair_basic": ["postcondition_repair"],
    "postcondition_repair_expand": ["postcondition_repair"],  # 也使用 postcondition_repair 的样本
    "assertion_reasoning_pipeline": ["assertion_reasoning_pipeline"],
    "uselemma": ["uselemma"],
    "instantiate_exists": ["instantiate_exists"],
    "instantiate_forall": ["instantiate_forall"],
    "case_analysis": ["case_analysis"],
    "seqsetmap": ["seqsetmap"],
    "compute": ["compute"],
    "extensional_equality": ["extensional_equality"],
    "reveal_opaque": ["reveal_opaque"],
    "inductive_lemma": ["inductive_lemma"],
    "induction": ["induction"],
    "loopinv": ["loopinv"],
    "arithmetic_reasoning": ["nonlinear_arithmetic"],
    "add_trigger_assert": ["add_trigger_assert"],
    "precondition_repair": ["precondition_repair"],
    "type_repair": ["type_repair"],
}


def optimize_prompt_sgd(
    initial_prompt: str,
    samples: List[Dict],
    instruction_lm: dspy.LM,
    minibatch_size: Optional[int],
    target_context_tokens: int,
    num_iters: Optional[int],
    seed: int,
) -> Tuple[str, Dict]:
    """
    SGD 风格优化 prompt
    
    Returns:
        (优化后的 prompt, cost 统计)
    """
    rng = random.Random(seed)
    
    # 洗牌样本
    shuffled_samples = list(samples)
    rng.shuffle(shuffled_samples)
    
    # 如果没有指定 minibatch_size，自动计算以填满 context window
    if minibatch_size is None:
        # 估算每个样本的 token 数
        sample_tokens = [estimate_sample_tokens(s) for s in shuffled_samples]
        avg_tokens = sum(sample_tokens) / len(sample_tokens) if sample_tokens else 1000
        
        # 留出空间给：prompt template (~1000) + current instruction (~5000，可能会变长) + output buffer (~16000)
        available_tokens = target_context_tokens - 22000
        minibatch_size = max(1, int(available_tokens / avg_tokens))
        
        print(f"    自动计算 minibatch_size: {minibatch_size} (平均样本 ~{int(avg_tokens)} tokens, 目标 {target_context_tokens} tokens)")
    
    # 计算迭代次数（如果未指定，则遍历所有样本一次）
    if num_iters is None:
        num_iters = (len(shuffled_samples) + minibatch_size - 1) // minibatch_size
    
    print(f"    样本数: {len(samples)}, Minibatch: {minibatch_size}, 迭代数: {num_iters}")
    
    current_prompt = initial_prompt
    total_input_tokens = 0
    total_output_tokens = 0
    iteration_logs = []
    
    for iter_idx in range(num_iters):
        # 获取当前 minibatch（循环使用样本）
        start_idx = (iter_idx * minibatch_size) % len(shuffled_samples)
        end_idx = start_idx + minibatch_size
        
        if end_idx <= len(shuffled_samples):
            minibatch = shuffled_samples[start_idx:end_idx]
        else:
            # 跨边界时拼接
            minibatch = shuffled_samples[start_idx:] + shuffled_samples[:end_idx - len(shuffled_samples)]
        
        if len(minibatch) == 0:
            continue
        
        # 创建 base program
        base_program = dspy.Predict(SimpleSignature)
        base_program.signature = base_program.signature.with_instructions(current_prompt)
        
        # 调用 ProposeNewInstructionModule
        proposer = ProposeNewInstructionModule(
            base_program=base_program,
            instruction_lm=instruction_lm,
            dataset_with_feedback=minibatch,
            knowledgebase_qe=None,
        )
        
        try:
            output = proposer.compile()
            new_prompt = output["new_instruction"]
            full_prompt_to_lm = output.get("full_prompt_to_teacher_lm", "")
            
            # 记录 token 使用
            input_tokens = count_tokens(full_prompt_to_lm)
            output_tokens = count_tokens(new_prompt)
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            
            iteration_logs.append({
                "iteration": iter_idx + 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            
            current_prompt = new_prompt
            print(f"      Iter {iter_idx + 1}/{num_iters}: input={input_tokens:,} tokens")
            
        except Exception as e:
            print(f"      Iter {iter_idx + 1}/{num_iters}: 错误 - {e}")
            continue
    
    stats = {
        "num_samples": len(samples),
        "num_iterations": num_iters,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "iteration_logs": iteration_logs,
    }
    
    return current_prompt, stats


def main():
    parser = argparse.ArgumentParser(description="VeruSAGE GEPA-SGD Prompt Optimization")
    parser.add_argument("--traces_dir", required=True, help="包含 trace JSON 文件的目录（如 verusage/train_5）")
    parser.add_argument("--prompt_names", required=True, 
                        help="要优化的 prompt 名称列表，逗号分隔（如 postcondition_repair_basic,assertion_reasoning_pipeline）")
    parser.add_argument(
        "--prompt_dir",
        default=str(Path(__file__).resolve().parents[1] / "verusage" / "agents" / "prompts"),
                        help="原始 prompt 文件目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--minibatch_size", type=int, default=None, 
                        help="Minibatch 大小（默认自动计算以填满 context window）")
    parser.add_argument("--target_context_tokens", type=int, default=120000,
                        help="目标 context window 大小（默认 120000 tokens，留 buffer）")
    parser.add_argument("--num_iters", type=int, default=None, 
                        help="迭代次数（默认自动计算以遍历所有样本）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    traces_dir = Path(args.traces_dir)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析 prompt 名称列表
    prompt_names = [p.strip() for p in args.prompt_names.split(",") if p.strip()]
    
    print("=" * 60)
    print("VeruSAGE GEPA-SGD Prompt Optimization")
    print("=" * 60)
    print(f"Traces 目录: {traces_dir}")
    print(f"要优化的 prompts: {prompt_names}")
    print(f"Minibatch size: {args.minibatch_size}")
    print(f"输出目录: {output_dir}")
    
    # 设置 LM
    print("\n配置 LM...")
    lm = setup_lm()
    dspy.configure(lm=lm)
    
    # 使用 PROMPT_TO_ACTIONS 映射，如果没有定义则回退到 ACTION_TO_PROMPT 的反向映射
    prompt_to_actions = dict(PROMPT_TO_ACTIONS)  # 使用预定义的映射
    # 补充 ACTION_TO_PROMPT 的反向映射
    for action, prompt in ACTION_TO_PROMPT.items():
        if prompt not in prompt_to_actions:
            prompt_to_actions.setdefault(prompt, []).append(action)
    
    # 汇总 cost
    total_cost = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "prompts": {},
    }
    
    # 优化每个 prompt
    for prompt_name in prompt_names:
        print(f"\n{'='*50}")
        print(f"优化 Prompt: {prompt_name}")
        print(f"{'='*50}")
        
        # 加载原始 prompt
        prompt_path = prompt_dir / f"{prompt_name}.md"
        if not prompt_path.exists():
            print(f"  跳过（prompt 文件不存在: {prompt_path}）")
            continue
        
        initial_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"  初始 prompt: {count_tokens(initial_prompt)} tokens")
        
        # 找到对应的 action 类型
        actions = prompt_to_actions.get(prompt_name, [prompt_name])
        print(f"  对应的 actions: {actions}")
        
        # 收集所有相关样本
        all_samples = []
        for action in actions:
            samples = load_samples_for_action(traces_dir, action)
            print(f"    {action}: {len(samples)} 个样本")
            all_samples.extend(samples)
        
        if len(all_samples) == 0:
            print("  跳过（无样本）")
            continue
        
        # 运行 SGD 优化
        print("  开始优化...")
        optimized_prompt, stats = optimize_prompt_sgd(
            initial_prompt=initial_prompt,
            samples=all_samples,
            instruction_lm=lm,
            minibatch_size=args.minibatch_size,
            target_context_tokens=args.target_context_tokens,
            num_iters=args.num_iters,
            seed=args.seed,
        )
        
        # 保存优化后的 prompt
        out_path = output_dir / f"{prompt_name}.md"
        out_path.write_text(optimized_prompt.strip() + "\n", encoding="utf-8")
        print(f"  保存到: {out_path}")
        
        # 记录 cost
        total_cost["total_input_tokens"] += stats["total_input_tokens"]
        total_cost["total_output_tokens"] += stats["total_output_tokens"]
        total_cost["prompts"][prompt_name] = {
            "input_tokens": stats["total_input_tokens"],
            "output_tokens": stats["total_output_tokens"],
            "num_samples": stats["num_samples"],
            "num_iterations": stats["num_iterations"],
        }
        
        print(f"  Cost: {stats['total_input_tokens']:,} input + {stats['total_output_tokens']:,} output tokens")
    
    # 计算总花费（Claude Sonnet 4.5 pricing: $3/1M input, $15/1M output）
    input_cost = total_cost["total_input_tokens"] / 1_000_000 * 3.0
    output_cost = total_cost["total_output_tokens"] / 1_000_000 * 15.0
    total_usd = input_cost + output_cost
    
    total_cost["cost_usd"] = {
        "input": input_cost,
        "output": output_cost,
        "total": total_usd,
    }
    
    # 保存 cost 统计
    cost_path = output_dir / "cost_summary.json"
    with open(cost_path, "w") as f:
        json.dump(total_cost, f, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("Cost 汇总")
    print("=" * 60)
    print(f"Total Input Tokens:  {total_cost['total_input_tokens']:,}")
    print(f"Total Output Tokens: {total_cost['total_output_tokens']:,}")
    print(f"Estimated Cost:      ${total_usd:.4f}")
    print(f"  - Input:  ${input_cost:.4f}")
    print(f"  - Output: ${output_cost:.4f}")
    print(f"\n结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
