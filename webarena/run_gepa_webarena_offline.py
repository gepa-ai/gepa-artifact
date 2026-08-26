"""
WebArena 离线 GEPA Prompt 优化

使用方式:
    python run_gepa_webarena_offline.py \
        --prompt_json p_cot_id_actree_2s.json \
        --traces_dir traces/ \
        --output_json p_cot_id_actree_2s_optimized.json \
        --lm_model openai/gpt-4o

数据格式:
    - prompt_json: WebArena 的 prompt 配置文件（包含 intro, examples, template, meta_data）
    - traces_dir: trace 文件夹，每个 trace 是一个 JSON 文件
    - GEPA 会修改 intro 部分并返回新的 JSON
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Optional

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from gepa_artifact.gepa.instruction_proposal import ProposeNewInstructionModule


def load_prompt_json(path: Path) -> dict:
    """加载 WebArena prompt JSON 配置"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_traces(traces_dir: Path, max_traces: Optional[int] = None) -> list[dict]:
    """加载所有 trace 文件"""
    traces = []
    trace_files = sorted(traces_dir.glob("trace_*.json"))
    
    if max_traces:
        trace_files = trace_files[:max_traces]
    
    for trace_file in trace_files:
        with open(trace_file, "r", encoding="utf-8") as f:
            traces.append(json.load(f))
    
    return traces


def generate_feedback_for_trace(trace: dict) -> str:
    """
    根据 trace 的结果生成反馈
    
    你可以根据需要自定义这个函数，添加更详细的反馈逻辑
    """
    score = trace.get("score", 0.0)
    intent = trace.get("intent", "Unknown task")
    num_steps = trace.get("num_steps", 0)
    action_history = trace.get("action_history", [])
    
    if score >= 1.0:
        return f"Task completed successfully in {num_steps} steps. The agent correctly achieved the objective: {intent}"
    else:
        # 对于失败的 trace，生成更详细的反馈
        final_action = action_history[-1] if action_history else "None"
        return (
            f"Task failed (score={score}). Objective was: {intent}. "
            f"Agent took {num_steps} steps but did not achieve the goal. "
            f"Final action: {final_action}. "
            f"Consider: Did the agent understand the task correctly? "
            f"Did it navigate to the right pages? Did it extract the correct information?"
        )


def trace_to_gepa_sample(trace: dict, prompt_template: str) -> dict:
    """
    将 WebArena trace 转换为 GEPA 需要的 dataset_with_feedback 格式
    
    Args:
        trace: WebArena trace dict
        prompt_template: prompt 模板，用于构建完整输入
    
    Returns:
        GEPA 格式的样本 dict
    """
    steps = trace.get("steps", [])
    if not steps:
        return None
    
    # 取最后一步作为主要样本（通常是关键决策点）
    # 你也可以改成取所有步骤
    last_step = steps[-1]
    
    # 构建输入（模拟 WebArena 的输入格式）
    input_text = prompt_template.format(
        observation=last_step.get("observation", ""),
        url=last_step.get("url", ""),
        objective=trace.get("intent", ""),
        previous_action=trace.get("action_history", ["None"])[-2] if len(trace.get("action_history", [])) > 1 else "None"
    )
    
    # 模型输出
    output_text = last_step.get("raw_prediction", "")
    
    # 生成反馈
    feedback = generate_feedback_for_trace(trace)
    
    return {
        "inputs": {"input": input_text},
        "generated_output": {"response": output_text},
        "feedback": feedback,
        "score": trace.get("score", 0.0),
        "task_id": trace.get("task_id"),
    }


def traces_to_dataset(
    traces: list[dict], 
    prompt_template: str,
    filter_by_score: Optional[str] = None,
    max_samples: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> tuple[list[dict], int]:
    """
    将所有 traces 转换为 GEPA dataset
    
    Args:
        traces: trace 列表
        prompt_template: prompt 模板
        filter_by_score: 过滤条件 - "failed"（只用失败的）, "success"（只用成功的）, None（全部）
        max_samples: 最大样本数
        rng: 随机数生成器
    
    Returns:
        (GEPA 格式的 dataset, 采样前的数量)
    """
    dataset = []
    
    for trace in traces:
        sample = trace_to_gepa_sample(trace, prompt_template)
        if sample is None:
            continue
        
        # 根据分数过滤
        if filter_by_score == "failed" and sample["score"] >= 1.0:
            continue
        if filter_by_score == "success" and sample["score"] < 1.0:
            continue
        
        dataset.append(sample)
    
    total_before_sampling = len(dataset)
    
    # 随机采样
    if max_samples is not None and len(dataset) > max_samples:
        if rng is None:
            rng = random.Random(42)
        dataset = rng.sample(dataset, max_samples)
        print(f"[INFO] 采样: {total_before_sampling} -> {len(dataset)} (max_samples={max_samples})")
    
    return dataset, total_before_sampling


def run_gepa_optimization(
    current_prompt: str,
    dataset_with_feedback: list[dict],
    instruction_lm: dspy.LM,
) -> dict:
    """
    运行 GEPA 优化
    
    Args:
        current_prompt: 当前的 prompt（intro 部分）
        dataset_with_feedback: GEPA 格式的数据集
        instruction_lm: 用于生成新 prompt 的 LM
    
    Returns:
        ProposeNewInstructionModule 的输出
    """
    # 创建一个简单的 signature，用于承载 prompt
    class WebArenaSignature(dspy.Signature):
        """WebArena agent prompt"""
        input: str = dspy.InputField()
        response: str = dspy.OutputField()
    
    # 创建 base_program 并设置当前 prompt
    base_program = dspy.Predict(WebArenaSignature)
    base_program.signature = base_program.signature.with_instructions(current_prompt)
    
    # 创建 proposer 并运行
    proposer = ProposeNewInstructionModule(
        base_program=base_program,
        instruction_lm=instruction_lm,
        dataset_with_feedback=dataset_with_feedback,
        knowledgebase_qe=None,
    )
    
    output = proposer.compile()
    return output


def configure_lm(args) -> dspy.LM:
    """配置 LM"""
    if args.lm_config_json:
        config = json.loads(args.lm_config_json)
        lm = dspy.LM(**config)
    elif args.lm_config_path:
        with open(args.lm_config_path, "r") as f:
            config = json.load(f)
        lm = dspy.LM(**config)
    elif args.lm_model:
        api_key = os.environ.get(args.lm_api_key_env) if args.lm_api_key_env else None
        lm = dspy.LM(model=args.lm_model, api_key=api_key, temperature=0.7, max_tokens=8192)
    else:
        raise ValueError("必须指定 --lm_model, --lm_config_path 或 --lm_config_json 之一")
    
    return lm


def main():
    parser = argparse.ArgumentParser(description="WebArena 离线 GEPA Prompt 优化")
    
    # 输入输出
    parser.add_argument("--prompt_json", required=True, help="WebArena prompt JSON 文件路径")
    parser.add_argument("--traces_dir", required=True, help="trace 文件夹路径")
    parser.add_argument("--output_json", required=True, help="输出的优化后 prompt JSON 路径")
    
    # LM 配置
    parser.add_argument("--lm_model", default=None, help="LM 模型名称 (如 openai/gpt-4o, anthropic/claude-sonnet-4-5-20250929)")
    parser.add_argument("--lm_api_key_env", default="OPENAI_API_KEY", help="API key 环境变量名")
    parser.add_argument("--lm_config_path", default=None, help="LM 配置 JSON 文件路径")
    parser.add_argument("--lm_config_json", default=None, help="LM 配置 JSON 字符串")
    
    # 数据过滤
    parser.add_argument("--filter_by_score", choices=["failed", "success", "all"], default="failed",
                       help="根据分数过滤 traces: failed=只用失败的, success=只用成功的, all=全部")
    parser.add_argument("--max_samples", type=int, default=10, help="最大样本数（用于 GEPA）")
    parser.add_argument("--max_traces", type=int, default=None, help="最大加载的 trace 数")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dry_run", action="store_true", help="只打印数据集，不运行 GEPA")
    
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    
    # 1. 加载 prompt JSON
    prompt_json_path = Path(args.prompt_json)
    prompt_config = load_prompt_json(prompt_json_path)
    current_intro = prompt_config.get("intro", "")
    template = prompt_config.get("template", "OBSERVATION:\n{observation}\nURL: {url}\nOBJECTIVE: {objective}\nPREVIOUS ACTION: {previous_action}")
    
    print(f"[INFO] 加载 prompt 配置: {prompt_json_path}")
    print(f"[INFO] 当前 intro 长度: {len(current_intro)} 字符")
    
    # 2. 加载 traces
    traces_dir = Path(args.traces_dir)
    traces = load_traces(traces_dir, max_traces=args.max_traces)
    print(f"[INFO] 加载了 {len(traces)} 个 traces")
    
    # 统计成功/失败
    success_count = sum(1 for t in traces if t.get("score", 0) >= 1.0)
    failed_count = len(traces) - success_count
    print(f"[INFO] 成功: {success_count}, 失败: {failed_count}")
    
    # 3. 转换为 GEPA dataset
    filter_score = None if args.filter_by_score == "all" else args.filter_by_score
    dataset, total_before_sampling = traces_to_dataset(
        traces, 
        template, 
        filter_by_score=filter_score,
        max_samples=args.max_samples,
        rng=rng,
    )
    print(f"[INFO] 过滤后样本数: {total_before_sampling}, 采样后用于 GEPA: {len(dataset)}")
    
    if not dataset:
        print("[ERROR] 没有样本可用于优化!")
        return
    
    # 打印样本预览
    print("\n[INFO] 样本预览 (第一个):")
    print("-" * 60)
    sample = dataset[0]
    print(f"Task ID: {sample.get('task_id')}")
    print(f"Score: {sample.get('score')}")
    print(f"Feedback: {sample.get('feedback')[:200]}...")
    print("-" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] 跳过 GEPA 优化")
        return
    
    # 4. 配置 LM
    instruction_lm = configure_lm(args)
    dspy.configure(lm=instruction_lm)
    print(f"[INFO] 配置 LM: {args.lm_model or 'from config'}")
    
    # 5. 运行 GEPA
    print("\n[INFO] 运行 GEPA 优化...")
    output = run_gepa_optimization(
        current_prompt=current_intro,
        dataset_with_feedback=dataset,
        instruction_lm=instruction_lm,
    )
    
    new_intro = output["new_instruction"]
    print(f"\n[INFO] 新 intro 长度: {len(new_intro)} 字符")
    
    # 6. 更新 prompt config 并保存
    prompt_config["intro"] = new_intro
    
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompt_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] 优化后的 prompt 已保存到: {output_path}")
    
    # 7. 打印新旧 prompt 对比
    print("\n" + "=" * 60)
    print("原始 intro (前 500 字符):")
    print("=" * 60)
    print(current_intro[:500])
    print("\n" + "=" * 60)
    print("优化后 intro (前 500 字符):")
    print("=" * 60)
    print(new_intro[:500])


if __name__ == "__main__":
    main()
