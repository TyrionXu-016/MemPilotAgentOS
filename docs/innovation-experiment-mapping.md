# 创新点与实验设计映射

实验按“创新点 -> 系统机制 -> 可复核证据”组织，所有聚合结果均可回溯到 612 条离线逐样例记录。

## 创新点一：长期记忆、规划、执行与反馈闭环

系统通过 `MemoryManager`、Retriever、Planner、Skill Registry 和 Runtime 形成反馈写回闭环。18 组隐式任务在写回前的任务覆盖率为 0.00，写回后为 1.00，说明反馈记忆实际改变了下一轮计划，而不是只作为日志保存。

对应证据：`agentos-evaluation-feedback-loop.csv`。

## 创新点二：面向任务规划的分层长期记忆

偏好记忆负责交互风格，反馈/任务记忆负责任务项，场景记忆提供具身上下文。72 条 clean 样例中：

| Planner | 偏好匹配率 | 任务覆盖率 |
|---|---:|---:|
| no_memory | 0.50 | 0.50 |
| preference_only | 1.00 | 0.50 |
| feedback_only | 0.50 | 1.00 |
| layered_memory | 1.00 | 1.00 |

与 `layered_memory` 相比，缺失相应记忆层的 36 个样例发生配对差异，Holm 校正后 `p < 1.75e-10`。这将偏好记忆和反馈记忆的贡献分别隔离出来。

对应证据：`agentos-evaluation.csv`、`agentos-evaluation-significance.csv`。

## 创新点三：技能约束下的领域可执行规划

Planner 输出结构化 `PlanStep(skill, params)`，计划执行前必须通过 `SkillRegistry.validate_plan()`。四个确定性 Planner 的参数可执行率均为 1.00；完整分层记忆 Planner 在三类场景中的领域技能链正确率为 1.00。无记忆 Planner 在无法辨识隐式家庭服务目标时技能链正确率为 0.92，说明“参数合法”和“领域技能选择正确”是两个不同指标。

对应证据：`agentos-evaluation-statistics.csv` 和技能负向单元测试。

## 创新点四：冲突与噪声条件下的记忆选择

Retriever 结合用户隔离、类型、相关性、importance 和 confidence 排序。`layered_memory` 在 clean、无关噪声、低置信冲突和低重要度历史记录四种条件下，偏好匹配率、任务覆盖率和冲突选择准确率均为 1.00。

对应证据：`agentos-evaluation-robustness.csv`。

## DeepSeek 外部基线

系统已实现 `deepseek_no_memory` 和 `deepseek_layered_memory`、五轮重复、JSON 校验、技能校验、并发执行和断点恢复。当前仓库未配置 `DEEPSEEK_API_KEY`，因此离线结果中不包含 DeepSeek 数值，论文暂不能把该接口实现写成已验证结论。正式运行后应引用 `agentos-evaluation-deepseek.csv`、manifest 中的 system fingerprint 和多数表决显著性结果。
