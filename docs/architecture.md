# AgentOS Prototype Architecture

AgentOS implements the following closed loop:

```text
SQLite long-term memory
  -> deterministic retrieval and evidence ranking
  -> goal/memory planning decision
  -> domain skill plan validation
  -> simulated embodied execution
  -> feedback memory writeback
  -> next-round planning and benchmark export
```

## Core Runtime

- `MemoryManager` and `AgentOSStore` persist profile, preference, task, scene, and feedback memories.
- `MemoryRetriever` isolates memories by user and ranks them by lexical relevance, memory type, importance, and confidence.
- Deterministic planners combine explicit goal information with selected memory layers. `Plan.context`, `Plan.evidence`, and `Plan.used_memory_ids` make the decision trace evaluable.
- `SkillRegistry` validates every `PlanStep` before execution. Learning and family-education tasks use quiz skills; home-service tasks use context retrieval, assistance preparation, notification, and verification skills.
- `AgentOSRuntime` executes validated skills through `SimulatedRobotAdapter` and writes feedback back to memory.

## Benchmark Layer

The checked-in benchmark contains 72 cases: three scenario groups, six user profiles per group, and four goal-explicitness levels per profile. Each case declares expected style, task items, relevant memories, and required skills.

The runner produces clean, ablation, robustness, and feedback-loop records. Statistics include Wilson 95% confidence intervals, exact McNemar tests with Holm correction, and latency summaries. Every aggregate can be traced back to `agentos-evaluation-cases.csv`.

## DeepSeek Baseline

`DeepSeekPlanGenerator` uses the official OpenAI-compatible DeepSeek endpoint and JSON Output. `deepseek_no_memory` receives only the goal and skill contracts; `deepseek_layered_memory` additionally receives the top-five retrieved memories. Returned JSON is validated locally with Pydantic and then with `SkillRegistry`.

DeepSeek runs are optional and resumable through a JSONL cache. The experiment manifest records the model name, system fingerprint, prompt hash, token counts, dataset hash, and implementation commit. No DeepSeek result is claimed when the API experiment has not been run.
