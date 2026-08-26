"""
GEPA-SGD: 简化版 GEPA，纯 SGD 式 prompt 优化

流程：
1. 用初始 prompt 跑整个 trainset，生成所有 trace + feedback（只跑一次！）
2. 每次迭代：
   - 从已生成的 trace 中取一个 minibatch
   - Teacher LM 根据 minibatch 的 feedback 生成新 prompt
   - 更新 prompt
3. 迭代直到遍历完训练集

优势：
- Trainset 只需要跑一次（初始 prompt）
- 后续迭代只调用 Teacher LM
- 公平对比：trace 都是用初始 prompt 生成的
"""

import os
import json
import random
import traceback
from typing import List, Callable, Optional

import dspy
from dspy import Example

from .instruction_proposal import ProposeNewInstructionModule
from .gepa_utils import capture_module_trace_with_feedback


class GEPA_SGD(dspy.teleprompt.teleprompt.Teleprompter):
    def __init__(
        self,
        named_predictor_to_feedback_fn_map: dict[str, Callable],
        knowledgebase_qe,
        metric: Callable,
        logger,
        run_dir: str,
        num_iters: int,
        minibatch_size: int = 10,
        teacher_lm: dspy.LM = None,
        seed: int = 0,
        num_threads: int = None,
        failure_score: float = 0,
        perfect_score: float = 1,
        skip_perfect_score: bool = True,
        predictor_subset: Optional[List[str]] = None,
    ):
        """
        Args:
            named_predictor_to_feedback_fn_map: predictor 名称 -> feedback 函数的映射
            knowledgebase_qe: 知识库查询引擎（可为 None）
            metric: 评估函数
            logger: 日志记录器
            run_dir: 运行输出目录
            num_iters: 迭代次数（每次迭代用一个 minibatch）
            minibatch_size: 每个 minibatch 的样本数
            teacher_lm: Teacher LM（用于生成新 prompt）
            seed: 随机种子
            num_threads: 并行线程数
        """
        self.named_predictor_to_feedback_fn_map = named_predictor_to_feedback_fn_map
        self.knowledgebase_qe = knowledgebase_qe
        self.metric_fn = metric
        self.logger = logger
        self.run_dir = run_dir
        self.num_iters = num_iters
        self.minibatch_size = minibatch_size
        self.teacher_lm = teacher_lm
        self.seed = seed
        self.num_threads = num_threads or os.cpu_count()
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.predictor_subset = predictor_subset

        # Minibatch 采样状态
        self.rng = random.Random(seed)

        # 跟踪
        self.iteration_logs = []

    def compile(
        self,
        student: dspy.Module,
        trainset: List[Example],
        valset: List[Example] = None,  # 忽略，只用 trainset
    ) -> dspy.Module:
        """
        编译优化 program
        
        Returns:
            优化后的 program
        """
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 当前 program（会被迭代更新）
        current_program = student.deepcopy()
        
        # 获取所有 predictor 名称
        named_predictors = list(current_program.named_predictors())
        predictor_names = [name for name, _ in named_predictors]
        if self.predictor_subset:
            allowed = set()
            for name in self.predictor_subset:
                allowed.add(name)
                if not name.endswith(".predict"):
                    allowed.add(f"{name}.predict")
            predictor_names = [name for name in predictor_names if name in allowed]
            if len(predictor_names) == 0:
                self.logger.log("GEPA-SGD: Predictor subset is empty after filtering; returning original program.")
                return current_program
        
        self.logger.log("GEPA-SGD: Starting optimization")
        self.logger.log(f"  - Trainset size: {len(trainset)}")
        self.logger.log(f"  - Minibatch size: {self.minibatch_size}")
        self.logger.log(f"  - Num iterations: {self.num_iters}")
        self.logger.log(f"  - Predictors: {predictor_names}")

        # ============================================================
        # 阶段 1: 用初始 prompt 跑整个 trainset，生成所有 trace + feedback
        # ============================================================
        self.logger.log("GEPA-SGD: Generating traces for entire trainset with initial prompt...")
        
        # 存储每个 predictor 的所有 trace+feedback
        all_traces_per_predictor = {}  # predictor_name -> list of (example_idx, trace_with_feedback)
        
        for predictor_name in predictor_names:
            if predictor_name not in self.named_predictor_to_feedback_fn_map:
                self.logger.log(f"  Predictor {predictor_name} not in feedback map, skipping")
                continue
            
            # 获取 predictor module
            module = None
            for name, mod in current_program.named_predictors():
                if name == predictor_name:
                    module = mod
                    break
            
            feedback_func = self.named_predictor_to_feedback_fn_map[predictor_name]
            
            self.logger.log(f"  Generating traces for predictor: {predictor_name}")
            
            # 用初始 prompt 跑整个 trainset
            dataset_with_feedback, total_score, all_scores = capture_module_trace_with_feedback(
                module=module,
                full_program=current_program,
                evalset=trainset,
                metric_fn=self.metric_fn,
                logger=self.logger,
                gepa_state=None,
                skip_perfect_score=self.skip_perfect_score,
                perfect_score=self.perfect_score,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
                feedback_func=feedback_func,
                add_format_failure_as_feedback=False,
                num_threads=self.num_threads,
                rng=self.rng,  # 传递 RNG
            )
            
            if dataset_with_feedback is not None:
                all_traces_per_predictor[predictor_name] = {
                    "traces": dataset_with_feedback,  # list of trace+feedback
                    "scores": all_scores,
                    "total_score": total_score,
                    "module": module,
                }
                self.logger.log(f"  Generated {len(dataset_with_feedback)} traces, total score: {total_score}")
            else:
                self.logger.log(f"  No traces generated for {predictor_name}")
        
        self.logger.log("GEPA-SGD: Trace generation complete")
        
        # 保存初始 trace 生成的统计
        with open(os.path.join(self.run_dir, "initial_traces.json"), 'w') as f:
            json.dump({
                predictor_name: {
                    "num_traces": len(info["traces"]),
                    "total_score": info["total_score"],
                    "scores": info["scores"],
                }
                for predictor_name, info in all_traces_per_predictor.items()
            }, f, indent=2, default=str)
        
        # ============================================================
        # 阶段 2: 迭代优化 - 每次从 trace 中取 minibatch，生成新 prompt
        # ============================================================
        self.logger.log("GEPA-SGD: Starting iterative optimization...")
        
        # 洗牌训练集索引
        shuffled_ids = list(range(len(trainset)))
        self.rng.shuffle(shuffled_ids)
        
        predictor_update_idx = 0  # 轮流更新各个 predictor
        updatable_predictors = [p for p in predictor_names if p in all_traces_per_predictor]
        
        if len(updatable_predictors) == 0:
            self.logger.log("GEPA-SGD: No updatable predictors, returning original program")
            return current_program

        for iter_idx in range(self.num_iters):
            try:
                iter_log = {"iteration": iter_idx + 1}
                
                # 1. 选择 minibatch 索引
                base_idx = (iter_idx * self.minibatch_size) % len(shuffled_ids)
                end_idx = base_idx + self.minibatch_size
                
                if end_idx <= len(shuffled_ids):
                    minibatch_ids = shuffled_ids[base_idx:end_idx]
                else:
                    # 跨 epoch 边界，需要 reshuffle
                    minibatch_ids = shuffled_ids[base_idx:]
                    self.rng.shuffle(shuffled_ids)
                    minibatch_ids += shuffled_ids[:end_idx - len(shuffled_ids)]
                
                iter_log["minibatch_ids"] = minibatch_ids
                
                # 2. 选择要更新的 predictor
                predictor_name = updatable_predictors[predictor_update_idx % len(updatable_predictors)]
                predictor_update_idx += 1
                iter_log["predictor_name"] = predictor_name
                
                trace_info = all_traces_per_predictor[predictor_name]
                all_traces = trace_info["traces"]
                module = trace_info["module"]
                
                # 获取 module 在 named_predictors 中的索引
                module_idx = None
                for idx, (name, _) in enumerate(current_program.named_predictors()):
                    if name == predictor_name:
                        module_idx = idx
                        break
                
                # 3. 从预生成的 trace 中提取 minibatch
                # all_traces 可能因为 skip_perfect_score 而少于 trainset
                # 需要映射 minibatch_ids 到 all_traces 的索引
                minibatch_traces = []
                for sample_idx in minibatch_ids:
                    if sample_idx < len(all_traces):
                        minibatch_traces.append(all_traces[sample_idx])
                
                if len(minibatch_traces) == 0:
                    self.logger.log(f"Iteration {iter_idx + 1}: No traces for minibatch, skipping")
                    self.iteration_logs.append(iter_log)
                    continue
                
                minibatch_score = sum(1 for t in minibatch_traces if t.get('score', 0) > 0)
                iter_log["minibatch_score"] = minibatch_score
                self.logger.log(f"Iteration {iter_idx + 1}: Minibatch score = {minibatch_score}/{len(minibatch_traces)}")
                
                # 4. Teacher LM 生成新 prompt（使用预生成的 trace）
                instruction_lm = self.teacher_lm or dspy.settings.lm or current_program.get_lm()
                self.logger.log(f"Iteration {iter_idx + 1}: Proposing new instruction for {predictor_name} using Teacher LM: {instruction_lm.model}")
                
                instruction_propose_module = ProposeNewInstructionModule(
                    base_program=module,
                    instruction_lm=instruction_lm,  # LM 在构造函数中设置
                    dataset_with_feedback=minibatch_traces,  # 使用预生成的 trace
                    knowledgebase_qe=self.knowledgebase_qe,
                )
                
                try:
                    output = instruction_propose_module.compile()
                    new_instruction = output['new_instruction']
                    kb_info = output.get('kb_info', None)
                    full_prompt = output.get('full_prompt_to_teacher_lm', '')  # 完整 prompt
                    
                    # 保存 instruction proposal 日志（包含完整输入以便计算 cost）
                    with open(os.path.join(self.run_dir, "instruction_proposals.jsonl"), 'a') as f:
                        f.write(json.dumps({
                            "iteration": iter_idx + 1,
                            "predictor": predictor_name,
                            "old_instruction": str(module.signature.instructions),
                            "new_instruction": new_instruction,
                            "kb_info": kb_info,
                            "minibatch_score": minibatch_score,
                            "minibatch_size": len(minibatch_traces),
                            # 保存完整 prompt 以便精确计算 Teacher LM cost
                            "full_prompt_to_teacher_lm": full_prompt,
                        }) + "\n")
                        
                except Exception as e:
                    self.logger.log(f"Iteration {iter_idx + 1}: Exception during instruction proposal: {e}")
                    self.logger.log(traceback.format_exc())
                    self.iteration_logs.append(iter_log)
                    continue
                
                iter_log["new_instruction"] = new_instruction
                self.logger.log(f"Iteration {iter_idx + 1}: New instruction: {new_instruction[:100]}...")
                
                # 5. 更新 prompt
                current_program.named_predictors()[module_idx][1].signature = \
                    module.signature.with_instructions(new_instruction)
                
                # 同时更新 module 引用（因为后续迭代的 ProposeNewInstructionModule 需要看到新 prompt）
                module.signature = module.signature.with_instructions(new_instruction)
                
                self.logger.log(f"Iteration {iter_idx + 1}: Updated prompt for {predictor_name}")
                
                self.iteration_logs.append(iter_log)
                
            except Exception as e:
                self.logger.log(f"Iteration {iter_idx + 1}: Exception: {e}")
                self.logger.log(traceback.format_exc())
                continue
        
        # 保存最终日志
        with open(os.path.join(self.run_dir, "gepa_sgd_log.json"), 'w') as f:
            json.dump(self.iteration_logs, f, indent=2, default=str)
        
        self.logger.log(f"GEPA-SGD: Optimization complete after {self.num_iters} iterations")
        
        return current_program
