"""
GEPA-SGD 可扩展性实验

目标：证明随着训练集大小增加，cost 会快速上升，同时效果不如 GEPA

实验设计：
1. training_trace 大小变化: [1, 2, 3, 5, 10, 50, 100, 150, 200, 300, 453]
2. GEPA-SGD 使用 minibatch 看完所有 segment
3. 只改同一个 prompt，迭代优化
4. trace 只用最开始的初始 prompt 生成（不重新跑）

记录指标：
- 每个训练集大小的 Teacher LM 调用 cost（输入 token 数）
- 最终验证集 accuracy
- 优化耗时
"""

import os
import sys
import json
import time
import dspy
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_metas
from gepa_artifact.gepa.gepa_sgd import GEPA_SGD
from gepa_artifact.gepa.gepa import GEPA
from gepa_artifact.utils.capture_stream_logger import Logger

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

# ============ 实验配置 ============
EXPERIMENT_CONFIG = {
    "trainset_sizes": [1, 2, 3, 5, 10, 50, 100, 150, 200, 300, 453],
    "minibatch_size": 10,  # GEPA-SGD 的 minibatch 大小
    "seed": 42,
    "num_threads": 4,
    "valset_size": 50,  # 固定验证集大小
}

# GEPA 基线配置（用于对比）
GEPA_CONFIG = {
    "num_dspy_examples_per_gepa_step": 3,
    "num_iters": 5,  # 固定迭代次数
}


def setup_azure_lm():
    """配置 Azure OpenAI LM"""
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


def count_tokens_in_prompt(prompt_text: str) -> int:
    """
    估算 prompt 的 token 数（使用简单估算：1 token ≈ 4 chars）
    更精确的方法可以使用 tiktoken
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(prompt_text))
    except ImportError:
        # 简单估算
        return len(prompt_text) // 4


def plot_scalability_results(results: Dict[str, Any], output_dir: Path):
    """绘制可扩展性实验结果图表"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return
    
    experiments = results["experiments"]
    if len(experiments) == 0:
        return
    
    # 提取数据
    trainset_sizes = [e["trainset_size"] for e in experiments]
    val_scores = [e["val_score"] for e in experiments]
    input_tokens = [e["total_input_tokens"] for e in experiments]
    times = [e["elapsed_time_seconds"] for e in experiments]
    base_score = results.get("base_score", 0)
    
    # 创建 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GEPA-SGD Scalability Analysis', fontsize=14, fontweight='bold')
    
    # 1. Val Score vs Trainset Size
    ax1 = axes[0, 0]
    ax1.plot(trainset_sizes, val_scores, 'b-o', linewidth=2, markersize=8, label='GEPA-SGD')
    ax1.axhline(y=base_score, color='r', linestyle='--', label=f'Baseline ({base_score:.2%})')
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Accuracy vs Training Set Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Input Tokens (Cost) vs Trainset Size
    ax2 = axes[0, 1]
    ax2.plot(trainset_sizes, input_tokens, 'g-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('Total Input Tokens to Teacher LM')
    ax2.set_title('Cost (Tokens) vs Training Set Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # 添加 trend line 展示增长速度
    if len(trainset_sizes) >= 3:
        log_sizes = np.log(trainset_sizes)
        log_tokens = np.log([max(t, 1) for t in input_tokens])
        slope, intercept = np.polyfit(log_sizes, log_tokens, 1)
        ax2.text(0.05, 0.95, f'Growth rate: O(n^{slope:.2f})', 
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Time vs Trainset Size
    ax3 = axes[1, 0]
    ax3.plot(trainset_sizes, times, 'm-^', linewidth=2, markersize=8)
    ax3.set_xlabel('Training Set Size')
    ax3.set_ylabel('Elapsed Time (seconds)')
    ax3.set_title('Time vs Training Set Size')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Cost-Effectiveness: Score per 1M tokens
    ax4 = axes[1, 1]
    cost_effectiveness = [(val_scores[i] - base_score) * 1_000_000 / max(input_tokens[i], 1) 
                          for i in range(len(experiments))]
    ax4.bar(range(len(trainset_sizes)), cost_effectiveness, color='coral', alpha=0.7)
    ax4.set_xticks(range(len(trainset_sizes)))
    ax4.set_xticklabels([str(s) for s in trainset_sizes], rotation=45)
    ax4.set_xlabel('Training Set Size')
    ax4.set_ylabel('Score Improvement per 1M Tokens')
    ax4.set_title('Cost-Effectiveness')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'scalability_analysis.pdf', bbox_inches='tight')
    print(f"图表已保存到: {output_dir / 'scalability_analysis.png'}")
    plt.close()
    
    # 生成 cost breakdown 表格
    with open(output_dir / 'cost_breakdown.md', 'w') as f:
        f.write("# GEPA-SGD Cost Breakdown\n\n")
        f.write("| Trainset Size | Iterations | Input Tokens | Est. Cost ($) | Val Score | Improvement |\n")
        f.write("|---------------|------------|--------------|---------------|-----------|-------------|\n")
        for i, e in enumerate(experiments):
            improvement = e["val_score"] - base_score
            est_cost = (e["total_input_tokens"] / 1_000_000) * 2.5
            f.write(f"| {e['trainset_size']} | {e['num_iterations']} | {e['total_input_tokens']:,} | ${est_cost:.4f} | {e['val_score']:.2%} | {improvement:+.2%} |\n")
        f.write(f"\n\nBaseline Score: {base_score:.2%}\n")


def load_benchmark_data():
    """加载 HotpotQA benchmark 数据"""
    print("正在加载 HotpotQA benchmark...")
    bench = hotpot_metas[0].benchmark()
    program = hotpot_metas[0].program[0]
    metric = hotpot_metas[0].metric
    feedback_fn_map = hotpot_metas[0].feedback_fn_maps[0]
    
    return bench, program, metric, feedback_fn_map


def estimate_gepa_sgd_cost(
    trainset_size: int,
    minibatch_size: int,
    traces_per_sample: int = 1,
    avg_tokens_per_trace: int = 500,  # 估算每个 trace 的平均 token 数
) -> Dict[str, Any]:
    """
    估算 GEPA-SGD 在给定训练集大小下的 cost
    
    Returns:
        dict with:
        - num_iterations: 迭代次数
        - total_input_tokens: 总输入 token 数（to Teacher LM）
        - cost_usd: 估算美元成本（GPT-4o pricing）
    """
    import math
    
    # 计算需要多少次迭代才能遍历整个训练集
    num_iterations = math.ceil(trainset_size / minibatch_size)
    
    # 每次迭代，Teacher LM 需要看 minibatch_size 个 trace
    # 随着迭代，累积的 trace context 会增长
    # GEPA-SGD 的特点：每次只看当前 minibatch，但 prompt 会逐步 accumulate feedback
    
    total_tokens = 0
    for iter_idx in range(num_iterations):
        # 当前 minibatch 的 token 数
        current_minibatch_size = min(minibatch_size, trainset_size - iter_idx * minibatch_size)
        minibatch_tokens = current_minibatch_size * avg_tokens_per_trace
        
        # 加上 prompt template 和当前 instruction 的开销
        overhead_tokens = 500  # 固定开销
        
        total_tokens += minibatch_tokens + overhead_tokens
    
    # GPT-4o pricing: ~$2.5 / 1M input tokens, $10 / 1M output tokens
    cost_usd = (total_tokens / 1_000_000) * 2.5 + (num_iterations * 200 / 1_000_000) * 10
    
    return {
        "trainset_size": trainset_size,
        "num_iterations": num_iterations,
        "total_input_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
    }


def run_gepa_sgd_experiment(
    trainset: List[dspy.Example],
    valset: List[dspy.Example],
    program: dspy.Module,
    metric,
    feedback_fn_map: Dict,
    run_dir: str,
    lm: dspy.LM,
    config: Dict,
) -> Dict[str, Any]:
    """
    运行单次 GEPA-SGD 实验
    
    Returns:
        实验结果字典
    """
    os.makedirs(run_dir, exist_ok=True)
    
    logger = Logger(os.path.join(run_dir, "run_log.txt"))
    
    trainset_size = len(trainset)
    minibatch_size = config["minibatch_size"]
    
    # 计算迭代次数：遍历整个训练集
    import math
    num_iters = math.ceil(trainset_size / minibatch_size)
    
    logger.log(f"Running GEPA-SGD with trainset_size={trainset_size}, minibatch_size={minibatch_size}, num_iters={num_iters}")
    
    # 创建 GEPA-SGD optimizer
    optimizer = GEPA_SGD(
        named_predictor_to_feedback_fn_map=feedback_fn_map,
        knowledgebase_qe=None,
        metric=metric,
        logger=logger,
        run_dir=run_dir,
        num_iters=num_iters,
        minibatch_size=minibatch_size,
        teacher_lm=lm,
        seed=config["seed"],
        num_threads=config["num_threads"],
    )
    
    # 配置 DSPy
    dspy.configure(lm=lm)
    
    start_time = time.time()
    
    try:
        optimized_program = optimizer.compile(
            program.deepcopy(),
            trainset=trainset,
            valset=None,
        )
        success = True
        error_msg = None
    except Exception as e:
        optimized_program = program
        success = False
        error_msg = str(e)
        traceback.print_exc()
    
    elapsed_time = time.time() - start_time
    
    # 评估优化后的程序
    evaluate = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=config["num_threads"],
        display_progress=True,
        max_errors=100
    )
    
    try:
        val_score = evaluate(optimized_program)
    except Exception as e:
        val_score = 0.0
        error_msg = f"Evaluation error: {e}"
    
    # 计算 Teacher LM 的实际 token 使用量
    # 从 instruction_proposals.jsonl 读取
    total_input_tokens = 0
    instruction_log_path = os.path.join(run_dir, "instruction_proposals.jsonl")
    if os.path.exists(instruction_log_path):
        with open(instruction_log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    prompt_text = entry.get("full_prompt_to_teacher_lm", "")
                    total_input_tokens += count_tokens_in_prompt(prompt_text)
                except (json.JSONDecodeError, TypeError):
                    pass
    
    result = {
        "trainset_size": trainset_size,
        "minibatch_size": minibatch_size,
        "num_iterations": num_iters,
        "val_score": val_score,
        "elapsed_time_seconds": elapsed_time,
        "total_input_tokens": total_input_tokens,
        "success": success,
        "error_msg": error_msg,
    }
    
    # 保存结果
    with open(os.path.join(run_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def run_full_experiment():
    """运行完整的可扩展性实验"""
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(__file__).parent.parent / "experiment_runs_data" / f"gepa_sgd_scalability_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"实验结果将保存到: {experiment_dir}")
    
    # 设置 LM
    lm = setup_azure_lm()
    
    # 加载数据
    bench, program, metric, feedback_fn_map = load_benchmark_data()
    
    # 获取完整训练集和验证集
    full_trainset = bench.train_set
    valset = bench.val_set[:EXPERIMENT_CONFIG["valset_size"]]
    
    print(f"完整训练集大小: {len(full_trainset)}")
    print(f"验证集大小: {len(valset)}")
    
    # 评估基线
    print("\n评估基线程序...")
    dspy.configure(lm=lm)
    evaluate = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=EXPERIMENT_CONFIG["num_threads"],
        display_progress=True,
        max_errors=100
    )
    base_score = evaluate(program)
    print(f"基线得分: {base_score:.2%}")
    
    # 存储所有结果
    all_results = {
        "config": EXPERIMENT_CONFIG,
        "base_score": base_score,
        "experiments": [],
        "cost_estimates": [],
    }
    
    # 先估算所有配置的理论 cost
    print("\n估算理论 cost...")
    for trainset_size in EXPERIMENT_CONFIG["trainset_sizes"]:
        if trainset_size > len(full_trainset):
            continue
        cost_est = estimate_gepa_sgd_cost(
            trainset_size=trainset_size,
            minibatch_size=EXPERIMENT_CONFIG["minibatch_size"],
        )
        all_results["cost_estimates"].append(cost_est)
        print(f"  trainset_size={trainset_size}: {cost_est['num_iterations']} iters, "
              f"~{cost_est['total_input_tokens']:,} tokens, ${cost_est['estimated_cost_usd']:.4f}")
    
    # 运行实验
    print("\n开始运行实验...")
    for trainset_size in EXPERIMENT_CONFIG["trainset_sizes"]:
        if trainset_size > len(full_trainset):
            print(f"\n跳过 trainset_size={trainset_size}（超过可用数据）")
            continue
            
        print(f"\n{'='*60}")
        print(f"运行 GEPA-SGD with trainset_size={trainset_size}")
        print(f"{'='*60}")
        
        # 随机采样训练集
        random.seed(EXPERIMENT_CONFIG["seed"])
        trainset_subset = random.sample(list(full_trainset), trainset_size)
        
        run_dir = str(experiment_dir / f"trainset_{trainset_size}")
        
        result = run_gepa_sgd_experiment(
            trainset=trainset_subset,
            valset=valset,
            program=program,
            metric=metric,
            feedback_fn_map=feedback_fn_map,
            run_dir=run_dir,
            lm=lm,
            config=EXPERIMENT_CONFIG,
        )
        
        all_results["experiments"].append(result)
        
        print(f"完成: val_score={result['val_score']:.2%}, "
              f"time={result['elapsed_time_seconds']:.1f}s, "
              f"tokens={result['total_input_tokens']:,}")
    
    # 保存汇总结果
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n实验完成！结果保存到: {summary_path}")
    
    # 绘制结果图表
    plot_scalability_results(all_results, experiment_dir)
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    print(f"{'Trainset Size':<15} {'Val Score':<12} {'Time (s)':<12} {'Input Tokens':<15} {'Iterations':<10}")
    print("-"*80)
    for result in all_results["experiments"]:
        print(f"{result['trainset_size']:<15} {result['val_score']:.2%}{'':8} "
              f"{result['elapsed_time_seconds']:.1f}{'':8} "
              f"{result['total_input_tokens']:<15,} {result['num_iterations']:<10}")
    
    return all_results


def run_comparison_with_gepa():
    """
    对比实验：GEPA vs GEPA-SGD
    在相同 budget 下比较两者的效果
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(__file__).parent.parent / "experiment_runs_data" / f"gepa_vs_sgd_comparison_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"对比实验结果将保存到: {experiment_dir}")
    
    lm = setup_azure_lm()
    bench, program, metric, feedback_fn_map = load_benchmark_data()
    
    trainset = bench.train_set[:150]  # 使用 150 个训练样本
    valset = bench.val_set[:50]
    
    dspy.configure(lm=lm)
    
    # 评估基线
    evaluate = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=4,
        display_progress=True,
        max_errors=100
    )
    base_score = evaluate(program)
    
    results = {
        "base_score": base_score,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
        "gepa_sgd": None,
        "gepa": None,
    }
    
    # 1. 运行 GEPA-SGD
    print("\n运行 GEPA-SGD...")
    sgd_run_dir = str(experiment_dir / "gepa_sgd")
    os.makedirs(sgd_run_dir, exist_ok=True)
    
    sgd_logger = Logger(os.path.join(sgd_run_dir, "run_log.txt"))
    
    sgd_optimizer = GEPA_SGD(
        named_predictor_to_feedback_fn_map=feedback_fn_map,
        knowledgebase_qe=None,
        metric=metric,
        logger=sgd_logger,
        run_dir=sgd_run_dir,
        num_iters=15,  # 150 / 10 = 15
        minibatch_size=10,
        teacher_lm=lm,
        seed=42,
        num_threads=4,
    )
    
    start_time = time.time()
    sgd_program = sgd_optimizer.compile(program.deepcopy(), trainset=trainset)
    sgd_time = time.time() - start_time
    sgd_score = evaluate(sgd_program)
    
    results["gepa_sgd"] = {
        "val_score": sgd_score,
        "elapsed_time": sgd_time,
    }
    
    # 2. 运行 GEPA
    print("\n运行 GEPA...")
    gepa_run_dir = str(experiment_dir / "gepa")
    os.makedirs(gepa_run_dir, exist_ok=True)
    
    gepa_logger = Logger(os.path.join(gepa_run_dir, "run_log.txt"))
    
    gepa_optimizer = GEPA(
        named_predictor_to_feedback_fn_map=feedback_fn_map,
        knowledgebase_qe=None,
        metric=metric,
        logger=gepa_logger,
        run_dir=gepa_run_dir,
        run_linearized_gepa=True,
        use_merge=False,
        num_threads=4,
        num_iters=5,  # 相同的迭代预算
        num_dspy_examples_per_gepa_step=3,
        track_scores_on='val',
        seed=42,
    )
    
    start_time = time.time()
    gepa_program = gepa_optimizer.compile(program.deepcopy(), trainset=trainset, valset=valset)
    gepa_time = time.time() - start_time
    gepa_score = evaluate(gepa_program)
    
    results["gepa"] = {
        "val_score": gepa_score,
        "elapsed_time": gepa_time,
    }
    
    # 保存结果
    with open(experiment_dir / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"基线: {base_score:.2%}")
    print(f"GEPA-SGD: {sgd_score:.2%} (耗时 {sgd_time:.1f}s)")
    print(f"GEPA: {gepa_score:.2%} (耗时 {gepa_time:.1f}s)")
    print(f"GEPA 相对 GEPA-SGD 提升: {(gepa_score - sgd_score):.2%}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GEPA-SGD 可扩展性实验")
    parser.add_argument("--mode", choices=["scalability", "comparison", "estimate_only"], 
                        default="scalability", help="实验模式")
    parser.add_argument("--trainset-sizes", type=str, default=None,
                        help="自定义训练集大小，逗号分隔，如 '1,2,5,10'")
    
    args = parser.parse_args()
    
    if args.trainset_sizes:
        EXPERIMENT_CONFIG["trainset_sizes"] = [int(x) for x in args.trainset_sizes.split(",")]
    
    if args.mode == "scalability":
        run_full_experiment()
    elif args.mode == "comparison":
        run_comparison_with_gepa()
    elif args.mode == "estimate_only":
        # 仅估算 cost
        print("估算不同训练集大小下的 GEPA-SGD cost:")
        print(f"{'Trainset Size':<15} {'Iterations':<12} {'Input Tokens':<15} {'Est. Cost ($)':<12}")
        print("-"*55)
        for size in EXPERIMENT_CONFIG["trainset_sizes"]:
            est = estimate_gepa_sgd_cost(size, EXPERIMENT_CONFIG["minibatch_size"])
            print(f"{size:<15} {est['num_iterations']:<12} {est['total_input_tokens']:<15,} ${est['estimated_cost_usd']:.4f}")
