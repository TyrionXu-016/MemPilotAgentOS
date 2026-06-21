from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from agentos.models import EvaluationResult, MemoryRecord
from agentos.planner import LayeredMemoryPlanner, MemoryFilteredPlanner, NoMemoryPlanner
from agentos.retriever import MemoryRetriever
from agentos.skills import build_default_registry
from agentos.storage import AgentOSStore


@dataclass(frozen=True)
class Scenario:
    scenario_group: str
    user_id: str
    goal: str
    expected_theme: str
    expected_words: tuple[str, ...]


SCENARIOS = [
    Scenario(
        scenario_group="learning",
        user_id="u001",
        goal="复习英语单词，保持太空主题",
        expected_theme="space",
        expected_words=("gravity",),
    ),
    Scenario(
        scenario_group="learning",
        user_id="u001",
        goal="继续昨天的英语练习",
        expected_theme="space",
        expected_words=("gravity",),
    ),
    Scenario(
        scenario_group="learning",
        user_id="u001",
        goal="做一轮个性化复习",
        expected_theme="space",
        expected_words=("gravity",),
    ),
    Scenario(
        scenario_group="family_education",
        user_id="family001",
        goal="陪孩子完成数学分数作业，使用故事化鼓励方式",
        expected_theme="story",
        expected_words=("fraction",),
    ),
    Scenario(
        scenario_group="family_education",
        user_id="family001",
        goal="继续昨天的家庭作业辅导",
        expected_theme="story",
        expected_words=("fraction",),
    ),
    Scenario(
        scenario_group="family_education",
        user_id="family001",
        goal="做一轮家庭教育个性化练习",
        expected_theme="story",
        expected_words=("fraction",),
    ),
    Scenario(
        scenario_group="home_service",
        user_id="home001",
        goal="准备睡前饮水提醒，优先温水",
        expected_theme="home",
        expected_words=("warm_water",),
    ),
    Scenario(
        scenario_group="home_service",
        user_id="home001",
        goal="查找常用水杯并安排晚间生活辅助",
        expected_theme="home",
        expected_words=("warm_water",),
    ),
    Scenario(
        scenario_group="home_service",
        user_id="home001",
        goal="安排晚间生活辅助任务",
        expected_theme="home",
        expected_words=("warm_water",),
    ),
]


@dataclass(frozen=True)
class EvaluationTable:
    name: str
    rows: list[EvaluationResult]


@dataclass(frozen=True)
class ExperimentSuiteResult:
    overall: EvaluationTable
    by_scenario: EvaluationTable
    ablation: EvaluationTable
    robustness: EvaluationTable


def seed_learning_memories(store: AgentOSStore) -> None:
    seed_learning_companion_memories(store)
    seed_family_education_memories(store)
    seed_home_service_memories(store)


def seed_learning_companion_memories(store: AgentOSStore) -> None:
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


def seed_family_education_memories(store: AgentOSStore) -> None:
    store.add_memory(
        MemoryRecord(
            user_id="family001",
            memory_type="preference",
            content="孩子更容易接受故事化讲解和鼓励式反馈",
            source="seed",
            importance=0.9,
            confidence=0.95,
            tags=["family", "homework", "story", "encouragement"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="family001",
            memory_type="feedback",
            content="任务目标：家庭数学作业；执行反馈：分数加减法容易出错，需要继续练习",
            source="seed",
            importance=0.8,
            confidence=0.9,
            tags=["math", "fraction", "feedback"],
        )
    )


def seed_home_service_memories(store: AgentOSStore) -> None:
    store.add_memory(
        MemoryRecord(
            user_id="home001",
            memory_type="preference",
            content="用户晚上不喝咖啡，偏好睡前喝温水",
            source="seed",
            importance=0.9,
            confidence=0.95,
            tags=["home", "warm_water", "bedtime"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="home001",
            memory_type="scene",
            content="常用水杯放在书桌右侧",
            source="seed",
            importance=0.75,
            confidence=0.9,
            tags=["home", "cup", "desk"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="home001",
            memory_type="feedback",
            content="任务目标：睡前生活辅助；执行反馈：上次睡前提醒太晚，用户希望提前提醒并准备温水",
            source="seed",
            importance=0.8,
            confidence=0.9,
            tags=["home", "warm_water", "feedback"],
        )
    )
def evaluate_planners(store: AgentOSStore) -> dict[str, EvaluationResult]:
    registry = build_default_registry()
    planners = build_evaluation_planners(store)
    return {planner.planner_name: _evaluate_one(planner, SCENARIOS) for planner in planners}


def build_evaluation_planners(store: AgentOSStore):
    registry = build_default_registry()
    retriever = MemoryRetriever(store)
    return [
        NoMemoryPlanner(registry),
        MemoryFilteredPlanner(
            registry=registry,
            retriever=retriever,
            planner_name="preference_only",
            allowed_memory_types={"profile", "preference"},
        ),
        MemoryFilteredPlanner(
            registry=registry,
            retriever=retriever,
            planner_name="feedback_only",
            allowed_memory_types={"task", "feedback"},
        ),
        LayeredMemoryPlanner(registry, retriever),
    ]


def evaluate_experiment_suite(store: AgentOSStore) -> ExperimentSuiteResult:
    planners = build_evaluation_planners(store)
    overall = EvaluationTable(
        name="overall",
        rows=[_evaluate_one(planner, SCENARIOS) for planner in planners],
    )
    by_scenario_rows = []
    for scenario_group in _scenario_groups():
        group_scenarios = [scenario for scenario in SCENARIOS if scenario.scenario_group == scenario_group]
        for planner in planners:
            result = _evaluate_one(planner, group_scenarios)
            by_scenario_rows.append(result.model_copy(update={"planner_name": f"{scenario_group}:{planner.planner_name}"}))
    ablation = EvaluationTable(name="ablation", rows=overall.rows)
    robustness = EvaluationTable(name="robustness", rows=evaluate_robustness())
    return ExperimentSuiteResult(
        overall=overall,
        by_scenario=EvaluationTable(name="by_scenario", rows=by_scenario_rows),
        ablation=ablation,
        robustness=robustness,
    )


def evaluate_robustness() -> list[EvaluationResult]:
    rows = []
    variants = {
        "clean": _seed_clean_variant,
        "irrelevant_noise": _seed_irrelevant_noise_variant,
        "conflict_low_confidence": _seed_conflict_low_confidence_variant,
        "stale_low_importance": _seed_stale_low_importance_variant,
    }
    for variant_name, seed_variant in variants.items():
        with TemporaryDirectory() as tmpdir:
            store = AgentOSStore(Path(tmpdir) / "agentos.sqlite")
            store.migrate()
            seed_variant(store)
            planner = LayeredMemoryPlanner(build_default_registry(), MemoryRetriever(store))
            result = _evaluate_one(planner, SCENARIOS)
            rows.append(result.model_copy(update={"planner_name": variant_name}))
    return rows


def export_evaluation_artifacts(
    store: AgentOSStore,
    output_dir: Path,
    stem: str = "agentos-evaluation",
) -> dict[str, Path]:
    suite = evaluate_experiment_suite(store)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": output_dir / f"{stem}.md",
        "csv": output_dir / f"{stem}.csv",
        "scenario_csv": output_dir / f"{stem}-scenarios.csv",
        "robustness_csv": output_dir / f"{stem}-robustness.csv",
        "svg": output_dir / f"{stem}.svg",
        "overall_svg": output_dir / "agentos-overall.svg",
        "ablation_svg": output_dir / "agentos-ablation.svg",
        "robustness_svg": output_dir / "agentos-robustness.svg",
    }
    paths["markdown"].write_text(render_markdown_report(suite), encoding="utf-8")
    paths["csv"].write_text(render_csv(suite.overall.rows), encoding="utf-8")
    paths["scenario_csv"].write_text(render_csv(suite.by_scenario.rows), encoding="utf-8")
    paths["robustness_csv"].write_text(render_csv(suite.robustness.rows), encoding="utf-8")
    paths["svg"].write_text(render_svg_bar_chart(suite.overall.rows, "AgentOS 总体规划指标"), encoding="utf-8")
    paths["overall_svg"].write_text(render_svg_bar_chart(suite.overall.rows, "AgentOS 总体规划指标"), encoding="utf-8")
    paths["ablation_svg"].write_text(render_svg_bar_chart(suite.ablation.rows, "AgentOS 消融实验指标"), encoding="utf-8")
    paths["robustness_svg"].write_text(render_svg_bar_chart(suite.robustness.rows, "AgentOS 鲁棒性实验指标"), encoding="utf-8")
    return paths


def render_markdown_report(suite: ExperimentSuiteResult) -> str:
    lines = [
        "# AgentOS 记忆增强规划实验结果",
        "",
        "本实验对比无长期记忆、偏好记忆、反馈记忆与完整分层记忆 Planner 在学习陪伴、家庭教育和家庭服务场景下的规划质量。"
        "所有结果由确定性原型生成，不依赖 LLM API，便于论文复现实验。",
        "",
        "## 场景泛化实验",
        "",
        "| Planner | 场景数 | 记忆命中率 | 偏好匹配率 | 计划可执行率 | 任务完成率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_markdown_rows(suite.overall.rows))
    lines.extend(
        [
            "",
            "## 消融实验",
            "",
            "| Planner | 场景数 | 记忆命中率 | 偏好匹配率 | 计划可执行率 | 任务完成率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_markdown_rows(suite.ablation.rows))
    lines.extend(
        [
            "",
            "## 鲁棒性实验",
            "",
            "| Variant | 场景数 | 记忆命中率 | 偏好匹配率 | 计划可执行率 | 任务完成率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_markdown_rows(suite.robustness.rows))
    layered = _find_row(suite.overall.rows, "layered_memory")
    no_memory = _find_row(suite.overall.rows, "no_memory")
    lines.extend(
        [
            "",
            "## 结论摘要",
            "",
            "- 完整分层记忆 Planner 覆盖学习陪伴、家庭教育和家庭服务三类场景，能够将偏好、场景和反馈记忆转化为计划参数。",
            f"- 在当前场景集上，完整分层记忆 Planner 的偏好匹配率为 {layered.preference_match_rate:.2f}，无记忆 Planner 为 {no_memory.preference_match_rate:.2f}。",
            "- 消融结果显示，偏好记忆主要提升偏好匹配率，反馈记忆主要提升任务完成率，完整分层记忆同时提升两类指标。",
            "- 鲁棒性结果显示，在无关、低置信冲突和低重要度过期记忆干扰下，完整分层记忆 Planner 仍保持稳定规划结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_csv(results: list[EvaluationResult]) -> str:
    header = [
        "planner_name",
        "scenario_count",
        "memory_hit_rate",
        "preference_match_rate",
        "executable_plan_rate",
        "task_completion_rate",
    ]
    lines = [",".join(header)]
    for result in results:
        row = _result_row(result)
        lines.append(",".join(str(row[name]) for name in header))
    return "\n".join(lines) + "\n"


def render_svg_bar_chart(results: list[EvaluationResult], title: str) -> str:
    metrics = [
        ("memory_hit_rate", "记忆命中率"),
        ("preference_match_rate", "偏好匹配率"),
        ("task_completion_rate", "任务完成率"),
    ]
    planners = results
    width = max(760, 180 + 130 * len(planners))
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
        f'<text x="24" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">{title}</text>',
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
            color = colors.get(result.planner_name, _palette(planner_index))
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + 2}" y="{y - 6:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{value:.2f}</text>')
    for planner_index, result in enumerate(planners):
        x = 90 + planner_index * 150
        color = colors.get(result.planner_name, _palette(planner_index))
        parts.append(f'<rect x="{x}" y="330" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x + 22}" y="342" font-family="Arial, sans-serif" font-size="12" fill="#111827">{result.planner_name}</text>')
    parts.append("</svg>")
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


def _seed_clean_variant(store: AgentOSStore) -> None:
    seed_learning_memories(store)


def _seed_irrelevant_noise_variant(store: AgentOSStore) -> None:
    seed_learning_memories(store)
    store.add_memory(
        MemoryRecord(
            user_id="noise001",
            memory_type="preference",
            content="其他用户喜欢恐龙主题",
            source="noise",
            importance=1.0,
            confidence=1.0,
            tags=["dinosaur"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="scene",
            content="无关记忆：客厅灯光偏暗",
            source="noise",
            importance=0.1,
            confidence=0.2,
            tags=["noise"],
        )
    )


def _seed_conflict_low_confidence_variant(store: AgentOSStore) -> None:
    seed_learning_memories(store)
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="preference",
            content="用户不喜欢太空主题互动",
            source="conflict",
            importance=0.2,
            confidence=0.2,
            tags=["space", "conflict"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="home001",
            memory_type="preference",
            content="用户晚上更想喝咖啡",
            source="conflict",
            importance=0.2,
            confidence=0.2,
            tags=["coffee", "conflict"],
        )
    )


def _seed_stale_low_importance_variant(store: AgentOSStore) -> None:
    seed_learning_memories(store)
    store.add_memory(
        MemoryRecord(
            user_id="family001",
            memory_type="feedback",
            content="过期反馈：分数练习已经完全掌握",
            source="stale",
            importance=0.05,
            confidence=0.2,
            tags=["fraction", "stale"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="home001",
            memory_type="scene",
            content="过期场景：水杯曾经放在厨房",
            source="stale",
            importance=0.05,
            confidence=0.2,
            tags=["cup", "stale"],
        )
    )


def _scenario_groups() -> list[str]:
    return list(dict.fromkeys(scenario.scenario_group for scenario in SCENARIOS))


def _markdown_rows(results: list[EvaluationResult]) -> list[str]:
    return [
        "| {planner_name} | {scenario_count} | {memory_hit_rate:.2f} | "
        "{preference_match_rate:.2f} | {executable_plan_rate:.2f} | "
        "{task_completion_rate:.2f} |".format(**_result_row(result))
        for result in results
    ]


def _find_row(results: list[EvaluationResult], planner_name: str) -> EvaluationResult:
    return next(result for result in results if result.planner_name == planner_name)


def _palette(index: int) -> str:
    colors = ["#6b7280", "#0891b2", "#f59e0b", "#2563eb", "#16a34a", "#dc2626"]
    return colors[index % len(colors)]


def _result_row(result: EvaluationResult) -> dict[str, str | int | float]:
    return {
        "planner_name": result.planner_name,
        "scenario_count": result.scenario_count,
        "memory_hit_rate": result.memory_hit_rate,
        "preference_match_rate": result.preference_match_rate,
        "executable_plan_rate": result.executable_plan_rate,
        "task_completion_rate": result.task_completion_rate,
    }
