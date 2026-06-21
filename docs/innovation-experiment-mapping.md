# 创新点与实验设计映射

本文实验设计以“创新点 -> 系统机制 -> 实验证据”的链路组织，避免只展示若干 demo 场景。

## 创新点一：AgentOS 长期记忆-规划-执行闭环架构

**系统机制**：AgentOS 原型将 `MemoryManager`、`MemoryRetriever`、`Planner`、`SkillRegistry`、`AgentOSRuntime` 和 `RobotAdapter` 连接成闭环。Planner 生成结构化技能计划，Runtime 执行技能并把反馈写回长期记忆。

**实验证据**：反馈闭环实验对比 `before_feedback_writeback` 与 `after_feedback_writeback`。执行前系统只有偏好记忆，能够匹配太空主题但无法覆盖 `gravity` 复习目标；执行反馈写回后，下一轮计划能够命中 `gravity`，说明反馈记忆改变了后续规划。

## 创新点二：面向任务规划的分层长期记忆模型

**系统机制**：记忆被划分为 `profile`、`preference`、`task`、`scene` 和 `feedback`，并带有 `importance`、`confidence` 和 `tags` 字段。不同记忆类型映射到不同规划作用：偏好记忆影响主题和交互方式，反馈/任务记忆影响任务目标，场景记忆影响具身上下文。

**实验证据**：消融实验比较 `no_memory`、`preference_only`、`feedback_only` 和 `layered_memory`。结果显示 `preference_only` 只提升偏好匹配率，`feedback_only` 只提升任务完成率，`layered_memory` 同时提升两类指标。

## 创新点三：技能约束下的可执行任务规划

**系统机制**：Planner 输出不是自然语言步骤，而是 `PlanStep(skill, params)`。所有计划在执行前经过 `SkillRegistry.validate_plan()` 校验，确保技能存在且必填参数齐全。

**实验证据**：场景泛化实验中四类 Planner 的计划可执行率均为 1.00，说明不同记忆配置下生成的计划都满足技能约束。相关 negative tests 验证缺少必填参数的计划会被 Skill Registry 拒绝。

## 创新点四：记忆选择鲁棒性

**系统机制**：Retriever 结合用户隔离、关键词匹配、记忆类型、`importance` 和 `confidence` 进行排序；低置信、低重要度或其他用户的噪声记忆不会覆盖高置信相关记忆。

**实验证据**：鲁棒性实验加入无关记忆、低置信冲突记忆和低重要度过期记忆，`layered_memory` 仍保持偏好匹配率和任务完成率为 1.00。

## 论文写作建议

论文中应按上述顺序组织：先提出创新点，再说明系统机制，最后引用对应实验表格和图表。实验章节不要只写“做了三个场景”，而应明确每个实验验证哪一条创新点。
