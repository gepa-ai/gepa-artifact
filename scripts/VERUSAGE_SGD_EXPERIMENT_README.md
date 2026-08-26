# VeruSAGE GEPA-SGD 可扩展性实验

## 目标

证明随着训练集大小增加，GEPA-SGD 的 cost 会快速上升，同时效果不如 GEPA。

## 实验设计

1. **训练集大小**: [1, 2, 3, 5, 10]（对应 `train_1`, `train_2`, ..., `train_10` 文件夹）
2. **GEPA-SGD 流程**:
   - 用初始 prompt 生成所有 trace（已预生成）
   - Minibatch 迭代优化 prompt
   - 只改同一个 prompt，trace 不重新跑
3. **记录指标**:
   - 每个训练集大小的 Teacher LM 调用 cost（输入 token 数）
   - 累积 token 数随训练集增长的趋势
   - 优化耗时

## 运行方式

### 基本运行

```bash
cd /path/to/gepa-artifact
python3 scripts/run_verusage_sgd_scalability.py
```

### 指定参数

```bash
# 使用不同的 minibatch 大小
python3 scripts/run_verusage_sgd_scalability.py --minibatch_size 5

# 使用不同的 prompt
python3 scripts/run_verusage_sgd_scalability.py --prompt_name assertion_reasoning_pipeline

# 使用所有 action 类型的样本
python3 scripts/run_verusage_sgd_scalability.py --all_actions

# 只使用特定 action 类型
python3 scripts/run_verusage_sgd_scalability.py --action_filter postcondition_repair
```

## 数据结构

训练数据位于：
- `verusage/train_1/` (1 个 trace 文件)
- `verusage/train_2/` (2 个 trace 文件)
- `verusage/train_3/` (3 个 trace 文件)
- `verusage/train_5/` (5 个 trace 文件)
- `verusage/train_10/` (10 个 trace 文件)

每个 trace 文件包含多个 `sub_agent_calling_*` 段落，每个段落是一个独立的样本。

## 输出

实验结果保存在 `experiment_runs_data/verusage_sgd_scalability_YYYYMMDD_HHMMSS/`:

- `experiment_summary.json` - 汇总结果
- `scalability_analysis.png/pdf` - 可视化图表
- `report.md` - Markdown 报告
- `train_*/` - 每个训练集大小的详细日志

## 主要 Action 类型分布

从训练数据统计：
- `assertion_reasoning_pipeline`: 121 次
- `postcondition_repair`: 85 次
- `uselemma`: 22 次
- `instantiate_exists`: 19 次
- `decfailend_repair`: 18 次
- `case_analysis`: 17 次
- 其他...

## 可用的 Prompts

位于 `verusage/agents/prompts/`:
- `postcondition_repair_basic.md`
- `assertion_reasoning_pipeline.md`
- `uselemma.md`
- `instantiate_exists.md`
- `case_analysis.md`
- 等等...

## 预期结果

1. **Cost 增长**: 随着训练集大小增加，总 input tokens 线性（或更快）增长
2. **每个样本的 tokens**: 由于 minibatch 累积，每个样本平均消耗的 tokens 可能增加
3. **与 GEPA 对比**: GEPA 可以重用 trace，效率更高
