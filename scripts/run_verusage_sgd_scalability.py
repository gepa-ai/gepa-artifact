"""
GEPA-SGD 在 VeruSAGE 上的可扩展性实验

目标：证明随着训练集大小增加，GEPA-SGD 的 cost 会快速上升

实验设计：
1. training_trace 大小变化: [1, 2, 3, 5, 10]（对应 train_1, train_2, ... train_10 文件夹）
2. GEPA-SGD 使用 minibatch 看完所有 segment，只用初始 prompt 生成的 trace
3. 只改同一个 prompt，迭代优化
4. 记录每次 Teacher LM 调用的 token 使用量

记录指标：
- 每个训练集大小的 Teacher LM 调用 cost（输入 token 数）
- 累积 token 数随训练集增长的趋势
- 优化耗时
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dspy

from gepa_artifact.gepa.instruction_proposal import ProposeNewInstructionModule


class SimpleSignature(dspy.Signature):
    """简单的输入输出签名"""
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


def extract_json_block(text: str) -> dict | None:
    """从文本中提取 JSON 块"""
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
        cand = text[start : end + 1]
        try:
            return json.loads(cand)
        except Exception:
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
        segments.append({"name": name, "trace": trace, "key": key})
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
    }


def load_all_traces_from_dir(traces_dir: Path) -> List[Dict]:
    """从目录加载所有 trace 并转换为样本"""
    all_samples = []
    for trace_path in sorted(traces_dir.glob("*.json")):
        segments = collect_segments_from_trace(trace_path)
        for seg in segments:
            sample = build_sample_from_segment(seg)
            if sample:
                all_samples.append(sample)
    return all_samples


def setup_lm():
    """配置 LM（使用 Azure OpenAI）"""
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
    from openai import AzureOpenAI

    AZURE_OPENAI_ENDPOINT = "https://trapi.research.microsoft.com/gcr/shared"
    AZURE_DEPLOYMENT_NAME = "gpt-4o_2024-11-20"
    AZURE_API_VERSION = "2025-04-01-preview"

    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        "api://trapi/.default",
    )

    client = AzureOpenAI(
        azure_ad_token_provider=credential,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
        max_retries=3,
    )

    lm = dspy.LM(
        model=AZURE_DEPLOYMENT_NAME,
        client=client,
        temperature=0.7,
        max_tokens=4096
    )
    return lm


class VeruSAGE_SGD_Optimizer:
    """
    GEPA-SGD 风格的 VeruSAGE prompt 优化器
    
    特点：
    1. 用初始 prompt 跑所有 trace（已经预生成）
    2. Minibatch 迭代优化 prompt
    3. 记录所有 Teacher LM 调用的 cost
    """
    
    def __init__(
        self,
        instruction_lm: dspy.LM,
        minibatch_size: int = 3,
        seed: int = 42,
    ):
        self.instruction_lm = instruction_lm
        self.minibatch_size = minibatch_size
        self.rng = random.Random(seed)
        
        # 记录
        self.iteration_logs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def optimize_prompt(
        self,
        initial_prompt: str,
        all_samples: List[Dict],
        run_dir: str,
    ) -> Tuple[str, Dict]:
        """
        SGD 风格优化 prompt
        
        Args:
            initial_prompt: 初始 prompt 文本
            all_samples: 所有训练样本（来自 trace）
            run_dir: 输出目录
            
        Returns:
            (优化后的 prompt, 统计信息)
        """
        os.makedirs(run_dir, exist_ok=True)
        
        # 洗牌样本
        shuffled_samples = list(all_samples)
        self.rng.shuffle(shuffled_samples)
        
        # 计算迭代次数
        num_samples = len(shuffled_samples)
        num_iters = (num_samples + self.minibatch_size - 1) // self.minibatch_size
        
        print(f"  样本数: {num_samples}, Minibatch: {self.minibatch_size}, 迭代数: {num_iters}")
        
        current_prompt = initial_prompt
        
        for iter_idx in range(num_iters):
            # 获取当前 minibatch
            start_idx = iter_idx * self.minibatch_size
            end_idx = min(start_idx + self.minibatch_size, num_samples)
            minibatch = shuffled_samples[start_idx:end_idx]
            
            if len(minibatch) == 0:
                continue
            
            # 创建 base program
            base_program = dspy.Predict(SimpleSignature)
            base_program.signature = base_program.signature.with_instructions(current_prompt)
            
            # 调用 ProposeNewInstructionModule
            proposer = ProposeNewInstructionModule(
                base_program=base_program,
                instruction_lm=self.instruction_lm,
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
                
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                
                iter_log = {
                    "iteration": iter_idx + 1,
                    "minibatch_size": len(minibatch),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cumulative_input_tokens": self.total_input_tokens,
                    "prompt_length": len(new_prompt),
                }
                self.iteration_logs.append(iter_log)
                
                # 保存到文件
                with open(os.path.join(run_dir, "iteration_logs.jsonl"), "a") as f:
                    f.write(json.dumps(iter_log) + "\n")
                
                # 保存完整 prompt 用于 cost 分析
                with open(os.path.join(run_dir, "full_prompts.jsonl"), "a") as f:
                    f.write(json.dumps({
                        "iteration": iter_idx + 1,
                        "full_prompt_to_teacher_lm": full_prompt_to_lm,
                    }) + "\n")
                
                # 更新 prompt
                current_prompt = new_prompt
                
                print(f"    Iter {iter_idx + 1}/{num_iters}: input_tokens={input_tokens:,}, cumulative={self.total_input_tokens:,}")
                
            except Exception as e:
                print(f"    Iter {iter_idx + 1}/{num_iters}: 错误 - {e}")
                continue
        
        # 保存最终 prompt
        final_prompt_path = os.path.join(run_dir, "optimized_prompt.md")
        with open(final_prompt_path, "w") as f:
            f.write(current_prompt)
        
        stats = {
            "num_samples": num_samples,
            "num_iterations": num_iters,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "iteration_logs": self.iteration_logs,
        }
        
        return current_prompt, stats


def run_scalability_experiment():
    """运行可扩展性实验"""
    
    parser = argparse.ArgumentParser(description="VeruSAGE GEPA-SGD 可扩展性实验")
    parser.add_argument("--minibatch_size", type=int, default=3, help="Minibatch 大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--prompt_name", default="postcondition_repair_basic", help="要优化的 prompt 名称")
    parser.add_argument("--action_filter", default="postcondition_repair", 
                        help="只使用指定 action 类型的样本（如 postcondition_repair）")
    parser.add_argument("--all_actions", action="store_true", help="使用所有 action 类型的样本")
    args = parser.parse_args()
    
    # 实验配置
    project_root = Path(__file__).resolve().parents[1]
    verusage_dir = project_root / "verusage"
    train_dirs = ["train_1", "train_2", "train_3", "train_5", "train_10"]
    prompt_dir = verusage_dir / "agents" / "prompts"
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = project_root / "experiment_runs_data" / f"verusage_sgd_scalability_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("VeruSAGE GEPA-SGD 可扩展性实验")
    print("=" * 70)
    print(f"实验目录: {experiment_dir}")
    print(f"Minibatch size: {args.minibatch_size}")
    print(f"Prompt: {args.prompt_name}")
    
    # 加载初始 prompt
    prompt_path = prompt_dir / f"{args.prompt_name}.md"
    if not prompt_path.exists():
        print(f"错误: Prompt 文件不存在 - {prompt_path}")
        return
    
    initial_prompt = prompt_path.read_text(encoding="utf-8").strip()
    print(f"初始 prompt 长度: {len(initial_prompt)} chars, ~{count_tokens(initial_prompt)} tokens")
    
    # 设置 LM
    print("\n配置 LM...")
    lm = setup_lm()
    dspy.configure(lm=lm)
    
    # 存储所有结果
    all_results = {
        "config": {
            "minibatch_size": args.minibatch_size,
            "seed": args.seed,
            "prompt_name": args.prompt_name,
            "action_filter": args.action_filter if not args.all_actions else "all",
            "initial_prompt_tokens": count_tokens(initial_prompt),
        },
        "experiments": [],
    }
    
    # 运行实验
    for train_dir_name in train_dirs:
        traces_dir = verusage_dir / train_dir_name
        if not traces_dir.exists():
            print(f"\n跳过 {train_dir_name}（不存在）")
            continue
        
        print(f"\n{'='*60}")
        print(f"训练集: {train_dir_name}")
        print(f"{'='*60}")
        
        # 加载所有样本
        all_samples = load_all_traces_from_dir(traces_dir)
        print(f"  加载了 {len(all_samples)} 个样本（所有 action 类型）")
        
        # 按 action 类型过滤
        if not args.all_actions and args.action_filter:
            filtered_samples = [s for s in all_samples if s.get("action_name") == args.action_filter]
            print(f"  过滤后 {len(filtered_samples)} 个样本（action={args.action_filter}）")
            all_samples = filtered_samples
        
        if len(all_samples) == 0:
            print("  跳过（无样本）")
            continue
        
        # 创建运行目录
        run_dir = str(experiment_dir / train_dir_name)
        
        # 运行优化
        start_time = time.time()
        
        optimizer = VeruSAGE_SGD_Optimizer(
            instruction_lm=lm,
            minibatch_size=args.minibatch_size,
            seed=args.seed,
        )
        
        optimized_prompt, stats = optimizer.optimize_prompt(
            initial_prompt=initial_prompt,
            all_samples=all_samples,
            run_dir=run_dir,
        )
        
        elapsed_time = time.time() - start_time
        
        result = {
            "train_dir": train_dir_name,
            "num_samples": stats["num_samples"],
            "num_iterations": stats["num_iterations"],
            "total_input_tokens": stats["total_input_tokens"],
            "total_output_tokens": stats["total_output_tokens"],
            "elapsed_time_seconds": elapsed_time,
            "iteration_logs": stats["iteration_logs"],
        }
        all_results["experiments"].append(result)
        
        print(f"  完成: {stats['num_iterations']} 迭代, "
              f"{stats['total_input_tokens']:,} input tokens, "
              f"{elapsed_time:.1f}s")
    
    # 保存汇总结果
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"{'Train Dir':<12} {'Samples':<10} {'Iters':<8} {'Input Tokens':<15} {'Time (s)':<10}")
    print("-" * 60)
    
    for result in all_results["experiments"]:
        print(f"{result['train_dir']:<12} {result['num_samples']:<10} "
              f"{result['num_iterations']:<8} {result['total_input_tokens']:<15,} "
              f"{result['elapsed_time_seconds']:.1f}")
    
    # 绘制结果
    plot_results(all_results, experiment_dir)
    
    print(f"\n结果已保存到: {experiment_dir}")
    
    return all_results


def plot_results(results: Dict, output_dir: Path):
    """绘制实验结果"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return
    
    experiments = results["experiments"]
    if len(experiments) < 2:
        return
    
    # 提取数据
    train_sizes = [e["num_samples"] for e in experiments]
    total_tokens = [e["total_input_tokens"] for e in experiments]
    times = [e["elapsed_time_seconds"] for e in experiments]
    # 创建 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('GEPA-SGD Scalability on VeruSAGE', fontsize=14, fontweight='bold')
    
    # 1. Total Tokens vs Training Size
    ax1 = axes[0, 0]
    ax1.plot(train_sizes, total_tokens, 'b-o', linewidth=2, markersize=10)
    ax1.set_xlabel('Training Samples (Segments)')
    ax1.set_ylabel('Total Input Tokens to Teacher LM')
    ax1.set_title('Cost Growth with Training Size')
    ax1.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(train_sizes) >= 3:
        z = np.polyfit(train_sizes, total_tokens, 1)
        p = np.poly1d(z)
        ax1.plot(train_sizes, p(train_sizes), 'r--', alpha=0.5, label='Linear fit')
        ax1.legend()
    
    # 2. Tokens per Sample
    ax2 = axes[0, 1]
    tokens_per_sample = [total_tokens[i] / max(train_sizes[i], 1) for i in range(len(experiments))]
    ax2.bar(range(len(train_sizes)), tokens_per_sample, color='green', alpha=0.7)
    ax2.set_xticks(range(len(train_sizes)))
    ax2.set_xticklabels([str(s) for s in train_sizes])
    ax2.set_xlabel('Training Samples')
    ax2.set_ylabel('Tokens per Sample')
    ax2.set_title('Efficiency: Tokens per Training Sample')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Time vs Training Size
    ax3 = axes[1, 0]
    ax3.plot(train_sizes, times, 'm-s', linewidth=2, markersize=10)
    ax3.set_xlabel('Training Samples')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Time vs Training Size')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Token Growth (from iteration logs)
    ax4 = axes[1, 1]
    for exp in experiments:
        if "iteration_logs" in exp and exp["iteration_logs"]:
            cumulative = [log["cumulative_input_tokens"] for log in exp["iteration_logs"]]
            iters = list(range(1, len(cumulative) + 1))
            ax4.plot(iters, cumulative, '-o', label=f"train_{exp['num_samples']}", markersize=5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cumulative Input Tokens')
    ax4.set_title('Cumulative Token Usage per Iteration')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'scalability_analysis.pdf', bbox_inches='tight')
    print(f"图表已保存到: {output_dir / 'scalability_analysis.png'}")
    plt.close()
    
    # 生成 Markdown 报告
    with open(output_dir / 'report.md', 'w') as f:
        f.write("# GEPA-SGD Scalability Experiment on VeruSAGE\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Minibatch size: {results['config']['minibatch_size']}\n")
        f.write(f"- Prompt: {results['config']['prompt_name']}\n")
        f.write(f"- Initial prompt tokens: {results['config']['initial_prompt_tokens']}\n\n")
        f.write("## Results\n\n")
        f.write("| Train Dir | Samples | Iterations | Input Tokens | Time (s) |\n")
        f.write("|-----------|---------|------------|--------------|----------|\n")
        for exp in experiments:
            f.write(f"| {exp['train_dir']} | {exp['num_samples']} | {exp['num_iterations']} | "
                    f"{exp['total_input_tokens']:,} | {exp['elapsed_time_seconds']:.1f} |\n")
        f.write("\n## Key Observations\n\n")
        f.write("1. **Cost Growth**: As training size increases, total input tokens grow.\n")
        f.write("2. **Linear Scaling**: Each iteration processes a minibatch, so cost scales with iterations.\n")
        f.write("3. **No Trace Reuse**: Unlike GEPA, GEPA-SGD cannot reuse traces across iterations.\n")


if __name__ == "__main__":
    run_scalability_experiment()
