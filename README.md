# MemPilot AgentOS Prototype

This repository contains a deterministic embodied AgentOS prototype for the thesis topic:

> 融合长期记忆与任务规划的具身智能 AgentOS 设计与实现

The prototype demonstrates a closed loop:

```text
long-term memory -> memory retrieval -> layered planning -> skill validation
-> simulated embodied execution -> feedback memory update -> evaluation export
```

The first version intentionally does not call an LLM API or real robot hardware. This keeps the experiment reproducible and makes the memory-planning mechanism easy to test.

## What It Includes

- Long-term memory storage with SQLite.
- Memory retrieval by user, task keywords, tags, importance, and confidence.
- Four planners:
  - `no_memory`: plans only from the current goal.
  - `preference_only`: retrieves only profile/preference memories.
  - `feedback_only`: retrieves only task/feedback memories.
  - `layered_memory`: retrieves long-term memory before planning.
- Skill Registry validation for executable structured plans.
- Simulated Robot Adapter for embodied terminal output.
- Evaluation scenarios for:
  - learning companion tasks
  - family education and homework tutoring tasks
  - home service and life-assistance tasks
- Paper-ready exports:
  - Markdown experiment report
  - CSV metrics tables
  - SVG charts

## Setup

Use Python 3.9 or newer.

```bash
python3 -m pip install -e ".[dev]"
```

## Run Tests

```bash
python3 -m pytest
```

Expected result:

```text
19 passed
```

## Run the Demo

Seed demo memories:

```bash
python3 -m agentos.cli seed
```

Run a memory-enhanced plan:

```bash
python3 -m agentos.cli run "复习英语单词，保持太空主题"
```

The generated plan should include:

```text
theme=space
words=["gravity"]
```

## Export Paper Results

```bash
python3 -m agentos.cli export-results \
  --db data/agentos-results.sqlite \
  --output-dir docs/results \
  --stem agentos-evaluation
```

Generated files:

- `docs/results/agentos-evaluation.md`
- `docs/results/agentos-evaluation.csv`
- `docs/results/agentos-evaluation.svg`
- `docs/results/agentos-evaluation-scenarios.csv`
- `docs/results/agentos-evaluation-robustness.csv`
- `docs/results/agentos-overall.svg`
- `docs/results/agentos-ablation.svg`
- `docs/results/agentos-robustness.svg`

The command creates a temporary SQLite database when using the example above. Remove it after export if it is only used for generation:

```bash
rm -f data/agentos-results.sqlite data/agentos-results.sqlite-shm data/agentos-results.sqlite-wal
```

## Current Experiment Result

The current evaluation uses nine scenarios across learning companion, family education, and home service tasks.

| Planner | Scenarios | Memory Hit Rate | Preference Match Rate | Executable Plan Rate | Task Completion Rate |
|---|---:|---:|---:|---:|---:|
| `no_memory` | 9 | 0.00 | 0.00 | 1.00 | 0.00 |
| `preference_only` | 9 | 1.00 | 1.00 | 1.00 | 0.00 |
| `feedback_only` | 9 | 1.00 | 0.00 | 1.00 | 1.00 |
| `layered_memory` | 9 | 1.00 | 1.00 | 1.00 | 1.00 |

See `docs/paper-experiment-section.md` for the thesis experiment section draft.
