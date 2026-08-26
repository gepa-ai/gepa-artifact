"""
快速测试 GEPA-SGD 可扩展性实验

使用小规模数据快速验证实验流程
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_metas
from gepa_artifact.gepa.gepa_sgd import GEPA_SGD
from gepa_artifact.utils.capture_stream_logger import Logger

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI


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


def count_tokens(text: str) -> int:
    """简单 token 计数"""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def run_quick_test():
    """运行快速测试"""
    
    print("=" * 60)
    print("GEPA-SGD 快速测试")
    print("=" * 60)
    
    # 设置实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(__file__).parent.parent / "experiment_runs_data" / f"gepa_sgd_quick_test_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"实验目录: {experiment_dir}")
    
    # 设置 LM
    print("\n1. 配置 LM...")
    lm = setup_azure_lm()
    dspy.configure(lm=lm)
    
    # 加载数据
    print("\n2. 加载 HotpotQA benchmark...")
    bench = hotpot_metas[0].benchmark()
    program = hotpot_metas[0].program[0]
    metric = hotpot_metas[0].metric
    feedback_fn_map = hotpot_metas[0].feedback_fn_maps[0]
    
    # 使用小规模数据
    trainset_sizes = [3, 5, 10]  # 小规模测试
    valset = bench.val_set[:10]  # 小验证集
    
    print(f"测试的训练集大小: {trainset_sizes}")
    print(f"验证集大小: {len(valset)}")
    
    # 评估基线
    print("\n3. 评估基线...")
    evaluate = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=4,
        display_progress=True,
        max_errors=100
    )
    base_score = evaluate(program)
    print(f"基线得分: {base_score:.2%}")
    
    # 运行实验
    results = []
    
    for trainset_size in trainset_sizes:
        print(f"\n{'='*50}")
        print(f"训练集大小: {trainset_size}")
        print(f"{'='*50}")
        
        random.seed(42)
        trainset = random.sample(list(bench.train_set), trainset_size)
        
        run_dir = str(experiment_dir / f"trainset_{trainset_size}")
        os.makedirs(run_dir, exist_ok=True)
        
        logger = Logger(os.path.join(run_dir, "run_log.txt"))
        
        # 计算迭代次数
        minibatch_size = 3
        num_iters = (trainset_size + minibatch_size - 1) // minibatch_size
        
        print(f"Minibatch size: {minibatch_size}, Iterations: {num_iters}")
        
        optimizer = GEPA_SGD(
            named_predictor_to_feedback_fn_map=feedback_fn_map,
            knowledgebase_qe=None,
            metric=metric,
            logger=logger,
            run_dir=run_dir,
            num_iters=num_iters,
            minibatch_size=minibatch_size,
            teacher_lm=lm,
            seed=42,
            num_threads=4,
        )
        
        start_time = time.time()
        optimized_program = optimizer.compile(
            program.deepcopy(),
            trainset=trainset,
        )
        elapsed_time = time.time() - start_time
        
        # 评估
        val_score = evaluate(optimized_program)
        
        # 读取 token 使用量
        total_tokens = 0
        log_path = os.path.join(run_dir, "instruction_proposals.jsonl")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        prompt = entry.get("full_prompt_to_teacher_lm", "")
                        total_tokens += count_tokens(prompt)
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        result = {
            "trainset_size": trainset_size,
            "num_iterations": num_iters,
            "val_score": val_score,
            "elapsed_time": elapsed_time,
            "total_tokens": total_tokens,
        }
        results.append(result)
        
        print(f"结果: val_score={val_score:.2%}, time={elapsed_time:.1f}s, tokens={total_tokens:,}")
    
    # 保存结果
    summary = {
        "base_score": base_score,
        "results": results,
    }
    
    with open(experiment_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"基线得分: {base_score:.2%}")
    print()
    print(f"{'Trainset':<10} {'Iters':<8} {'Val Score':<12} {'Time (s)':<10} {'Tokens':<12}")
    print("-" * 55)
    for r in results:
        print(f"{r['trainset_size']:<10} {r['num_iterations']:<8} {r['val_score']:.2%}{'':4} {r['elapsed_time']:.1f}{'':5} {r['total_tokens']:,}")
    
    print(f"\n结果已保存到: {experiment_dir}")
    
    return results


if __name__ == "__main__":
    run_quick_test()
