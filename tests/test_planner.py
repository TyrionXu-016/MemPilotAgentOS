from agentos.models import MemoryRecord
from agentos.planner import LayeredMemoryPlanner, NoMemoryPlanner
from agentos.retriever import MemoryRetriever
from agentos.skills import build_default_registry


def test_no_memory_planner_uses_generic_theme(store):
    planner = NoMemoryPlanner(build_default_registry())
    plan = planner.plan(user_id="u001", goal="复习英语单词")
    assert plan.planner_name == "no_memory"
    assert plan.steps[1].params["theme"] == "neutral"


def test_layered_memory_planner_uses_preference_and_feedback(store):
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="preference",
            content="用户喜欢太空主题互动",
            source="seed",
            importance=0.9,
            tags=["space"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="feedback",
            content="任务目标：复习英语单词；执行反馈：gravity 答错",
            source="runtime",
            importance=0.8,
            tags=["gravity"],
        )
    )
    planner = LayeredMemoryPlanner(build_default_registry(), MemoryRetriever(store))
    plan = planner.plan(user_id="u001", goal="复习英语单词，保持太空主题")
    assert plan.planner_name == "layered_memory"
    assert plan.steps[1].params["theme"] == "space"
    assert "gravity" in plan.steps[1].params["words"]
    assert any("太空" in basis for basis in plan.basis)
