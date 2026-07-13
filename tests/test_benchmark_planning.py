from pathlib import Path

import pytest

from agentos.benchmark.dataset import load_benchmark
from agentos.planner import LayeredMemoryPlanner, MemoryFilteredPlanner, NoMemoryPlanner
from agentos.retriever import MemoryRetriever
from agentos.skills import build_default_registry
from agentos.storage import AgentOSStore


def _plan_case(case, planner_name: str, db_path: Path):
    store = AgentOSStore(db_path)
    store.migrate()
    for memory in case.memories:
        store.add_memory(memory)
    registry = build_default_registry()
    if planner_name == "no_memory":
        planner = NoMemoryPlanner(registry)
    elif planner_name == "preference_only":
        planner = MemoryFilteredPlanner(
            registry,
            MemoryRetriever(store),
            planner_name="preference_only",
            allowed_memory_types={"profile", "preference"},
        )
    elif planner_name == "feedback_only":
        planner = MemoryFilteredPlanner(
            registry,
            MemoryRetriever(store),
            planner_name="feedback_only",
            allowed_memory_types={"task", "feedback"},
        )
    else:
        planner = LayeredMemoryPlanner(registry, MemoryRetriever(store))
    return planner.plan(case.user_id, case.goal)


@pytest.mark.parametrize(
    ("planner_name", "expected_style_rate", "expected_item_rate"),
    [
        ("no_memory", 0.5, 0.5),
        ("preference_only", 1.0, 0.5),
        ("feedback_only", 0.5, 1.0),
        ("layered_memory", 1.0, 1.0),
    ],
)
def test_deterministic_planners_have_balanced_benchmark_rates(
    tmp_path, planner_name, expected_style_rate, expected_item_rate
):
    cases = load_benchmark().cases
    plans = [
        _plan_case(case, planner_name, tmp_path / f"{planner_name}-{index}.sqlite")
        for index, case in enumerate(cases)
    ]

    style_rate = sum(
        plan.context.interaction_style == case.expected_style
        for plan, case in zip(plans, cases)
    ) / len(cases)
    item_rate = sum(
        bool(set(plan.context.task_items) & set(case.expected_items))
        for plan, case in zip(plans, cases)
    ) / len(cases)
    assert style_rate == expected_style_rate
    assert item_rate == expected_item_rate


def test_layered_home_service_plan_uses_domain_skill_chain(tmp_path):
    case = next(
        case
        for case in load_benchmark().cases
        if case.scenario_group == "home_service" and case.explicitness == "implicit"
    )
    plan = _plan_case(case, "layered_memory", tmp_path / "home.sqlite")

    assert plan.context.scenario_group == "home_service"
    assert [step.skill for step in plan.steps] == case.required_skills
    assert plan.evidence
    assert plan.used_memory_ids
    build_default_registry().validate_plan(plan)

