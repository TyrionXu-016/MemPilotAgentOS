from __future__ import annotations

from dataclasses import dataclass

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
