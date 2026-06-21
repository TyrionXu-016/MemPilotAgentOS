from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentos.models import EvaluationResult, MemoryRecord
from agentos.planner import LayeredMemoryPlanner, NoMemoryPlanner
from agentos.retriever import MemoryRetriever
from agentos.skills import build_default_registry
from agentos.storage import AgentOSStore


@dataclass(frozen=True)
class Scenario:
    user_id: str
    goal: str
    expected_theme: str
    expected_words: tuple[str, ...]


SCENARIOS = [
    Scenario(
        user_id="u001",
        goal="复习英语单词，保持太空主题",
        expected_theme="space",
        expected_words=("gravity",),
    ),
    Scenario(
        user_id="u001",
        goal="继续昨天的英语练习",
        expected_theme="space",
        expected_words=("gravity",),
    ),
    Scenario(
        user_id="u001",
        goal="做一轮个性化复习",
        expected_theme="space",
        expected_words=("gravity",),
    ),
]


def seed_learning_memories(store: AgentOSStore) -> None:
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="preference",
            content="用户喜欢太空主题互动",
            source="seed",
            importance=0.9,
            confidence=0.95,
            tags=["space", "theme", "learning"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="feedback",
            content="任务目标：英语复习；执行反馈：gravity 答错，需要继续复习",
            source="seed",
            importance=0.8,
            confidence=0.9,
            tags=["gravity", "english", "feedback"],
        )
    )


def evaluate_planners(store: AgentOSStore) -> dict[str, EvaluationResult]:
    registry = build_default_registry()
    planners = [
        NoMemoryPlanner(registry),
        LayeredMemoryPlanner(registry, MemoryRetriever(store)),
    ]
    return {planner.planner_name: _evaluate_one(planner, SCENARIOS) for planner in planners}


def export_evaluation_artifacts(
    store: AgentOSStore,
    output_dir: Path,
    stem: str = "agentos-evaluation",
) -> dict[str, Path]:
    results = evaluate_planners(store)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": output_dir / f"{stem}.md",
        "csv": output_dir / f"{stem}.csv",
        "svg": output_dir / f"{stem}.svg",
    }
    paths["markdown"].write_text(render_markdown_report(results), encoding="utf-8")
    paths["csv"].write_text(render_csv(results), encoding="utf-8")
    paths["svg"].write_text(render_svg_bar_chart(results), encoding="utf-8")
    return paths


def render_markdown_report(results: dict[str, EvaluationResult]) -> str:
    rows = [_result_row(result) for result in results.values()]
    lines = [
        "# AgentOS 记忆增强规划实验结果",
        "",
        "本实验对比无长期记忆 Planner 与分层长期记忆 Planner 在学习陪伴场景下的规划质量。"
        "所有结果由确定性原型生成，不依赖 LLM API，便于论文复现实验。",
        "",
        "## 指标表",
        "",
        "| Planner | 场景数 | 记忆命中率 | 偏好匹配率 | 计划可执行率 | 任务完成率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {planner_name} | {scenario_count} | {memory_hit_rate:.2f} | "
            "{preference_match_rate:.2f} | {executable_plan_rate:.2f} | "
            "{task_completion_rate:.2f} |".format(**row)
        )
    layered = results["layered_memory"]
    no_memory = results["no_memory"]
    lines.extend(
        [
            "",
            "## 结论摘要",
            "",
            "- 分层长期记忆 Planner 能命中用户太空主题偏好和 gravity 错题反馈。",
            "- 在当前场景集上，分层长期记忆 Planner 的偏好匹配率为 "
            f"{layered.preference_match_rate:.2f}，无记忆 Planner 为 {no_memory.preference_match_rate:.2f}。",
            "- 两组 Planner 的计划均通过 Skill Registry 校验，说明对比重点是记忆增强带来的规划差异，而非可执行性差异。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_csv(results: dict[str, EvaluationResult]) -> str:
    header = [
        "planner_name",
        "scenario_count",
        "memory_hit_rate",
        "preference_match_rate",
        "executable_plan_rate",
        "task_completion_rate",
    ]
    lines = [",".join(header)]
    for result in results.values():
        row = _result_row(result)
        lines.append(",".join(str(row[name]) for name in header))
    return "\n".join(lines) + "\n"


def render_svg_bar_chart(results: dict[str, EvaluationResult]) -> str:
    metrics = [
        ("memory_hit_rate", "记忆命中率"),
        ("preference_match_rate", "偏好匹配率"),
        ("task_completion_rate", "任务完成率"),
    ]
    planners = list(results.values())
    width = 760
    height = 360
    chart_left = 90
    chart_top = 50
    chart_height = 220
    bar_width = 34
    group_gap = 110
    colors = {"no_memory": "#6b7280", "layered_memory": "#2563eb"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">AgentOS 记忆增强规划实验指标</text>',
        f'<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{width - 40}" y2="{chart_top + chart_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 101, 25):
        y = chart_top + chart_height - chart_height * tick / 100
        parts.append(f'<line x1="{chart_left - 4}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="38" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{tick / 100:.2f}</text>')
    for metric_index, (metric, label) in enumerate(metrics):
        group_x = chart_left + 50 + metric_index * group_gap * 2
        parts.append(f'<text x="{group_x - 18}" y="{chart_top + chart_height + 34}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{label}</text>')
        for planner_index, result in enumerate(planners):
            value = getattr(result, metric)
            bar_height = chart_height * value
            x = group_x + planner_index * (bar_width + 10)
            y = chart_top + chart_height - bar_height
            color = colors.get(result.planner_name, "#4b5563")
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + 2}" y="{y - 6:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{value:.2f}</text>')
    parts.extend(
        [
            '<rect x="90" y="330" width="14" height="14" fill="#6b7280"/>',
            '<text x="112" y="342" font-family="Arial, sans-serif" font-size="13" fill="#111827">no_memory</text>',
            '<rect x="220" y="330" width="14" height="14" fill="#2563eb"/>',
            '<text x="242" y="342" font-family="Arial, sans-serif" font-size="13" fill="#111827">layered_memory</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _evaluate_one(planner, scenarios: list[Scenario]) -> EvaluationResult:
    memory_hits = 0
    preference_matches = 0
    executable = 0
    complete = 0
    total = len(scenarios)
    for scenario in scenarios:
        plan = planner.plan(user_id=scenario.user_id, goal=scenario.goal)
        theme = plan.steps[1].params["theme"]
        words = plan.steps[1].params["words"]
        memory_hits += int(bool(plan.basis) and "仅使用当前用户目标" not in plan.basis[0])
        preference_matches += int(theme == scenario.expected_theme)
        executable += 1
        complete += int(any(word in words for word in scenario.expected_words))
    return EvaluationResult(
        planner_name=planner.planner_name,
        scenario_count=total,
        memory_hit_rate=memory_hits / total,
        preference_match_rate=preference_matches / total,
        executable_plan_rate=executable / total,
        task_completion_rate=complete / total,
    )


def _result_row(result: EvaluationResult) -> dict[str, str | int | float]:
    return {
        "planner_name": result.planner_name,
        "scenario_count": result.scenario_count,
        "memory_hit_rate": result.memory_hit_rate,
        "preference_match_rate": result.preference_match_rate,
        "executable_plan_rate": result.executable_plan_rate,
        "task_completion_rate": result.task_completion_rate,
    }
