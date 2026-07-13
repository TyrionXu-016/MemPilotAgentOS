# MemPilot AgentOS Prototype

AgentOS is a reproducible software prototype for the thesis topic:

> 融合长期记忆与任务规划的具身智能 AgentOS 设计与实现

It implements a closed loop from SQLite long-term memory through retrieval, layered planning, skill validation, simulated embodied execution, feedback writeback, and experiment export.

## Included Experiments

- 72 balanced cases across learning companion, family education, and home service.
- Four deterministic planners: `no_memory`, `preference_only`, `feedback_only`, and `layered_memory`.
- 288 clean planner runs, 288 robustness runs, and 36 feedback-loop runs.
- Optional `deepseek_no_memory` and `deepseek_layered_memory` baselines with five repeats per case.
- Wilson confidence intervals, exact McNemar tests, Holm correction, latency metrics, raw case CSV, manifest, and SVG charts.

## Setup

Use Python 3.9 or newer:

```bash
python3 -m pip install -e ".[dev]"
```

Install the optional DeepSeek client dependency with:

```bash
python3 -m pip install -e ".[dev,llm]"
```

## Test And Demo

```bash
python3 -m pytest
python3 -m agentos.cli seed
python3 -m agentos.cli run "复习英语单词，保持太空主题"
```

The current suite contains 40 tests.

## Export Deterministic Results

```bash
python3 -m agentos.cli export-results \
  --db data/agentos-results.sqlite \
  --dataset experiments/data/agentos-benchmark-v1.json \
  --output-dir docs/results \
  --stem agentos-evaluation
```

The legacy `--db` option remains accepted, but benchmark cases use isolated temporary SQLite stores and do not leave the specified database behind.

## Run DeepSeek Baselines

```bash
DEEPSEEK_API_KEY=... python3 -m agentos.cli export-results \
  --dataset experiments/data/agentos-benchmark-v1.json \
  --output-dir docs/results \
  --stem agentos-evaluation \
  --include-deepseek \
  --deepseek-model deepseek-v4-flash \
  --deepseek-repeats 5 \
  --concurrency 5 \
  --cache .agentos-cache/deepseek-runs.jsonl
```

The cache is resumable and ignored by Git. Without `DEEPSEEK_API_KEY`, no DeepSeek result is generated or claimed.

## Current Deterministic Result

| Planner | Cases | Preference Match | Task Coverage | Executable | Skill Chain |
|---|---:|---:|---:|---:|---:|
| `no_memory` | 72 | 0.50 | 0.50 | 1.00 | 0.92 |
| `preference_only` | 72 | 1.00 | 0.50 | 1.00 | 1.00 |
| `feedback_only` | 72 | 0.50 | 1.00 | 1.00 | 1.00 |
| `layered_memory` | 72 | 1.00 | 1.00 | 1.00 | 1.00 |

See `docs/results/agentos-evaluation.md` for generated results, `docs/paper-experiment-section.md` for the thesis draft, and `docs/innovation-experiment-mapping.md` for the innovation-to-evidence mapping.
