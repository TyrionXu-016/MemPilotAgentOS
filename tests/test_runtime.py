from agentos.memory import MemoryManager
from agentos.models import Plan, PlanStep
from agentos.runtime import AgentOSRuntime
from agentos.skills import build_default_registry


def test_runtime_executes_plan_and_writes_feedback(store):
    runtime = AgentOSRuntime(build_default_registry(), MemoryManager(store))
    plan = Plan(
        goal="复习英语单词",
        basis=["用户喜欢太空主题"],
        steps=[
            PlanStep(skill="generate_quiz", params={"theme": "space", "words": ["gravity"]}),
            PlanStep(skill="ask_question", params={"mode": "interactive"}),
            PlanStep(skill="evaluate_answer", params={"expected_words": ["gravity"]}),
            PlanStep(skill="update_memory", params={"feedback": "gravity still needs practice"}),
        ],
    )
    events = runtime.run(user_id="u001", plan=plan)
    assert [event.status for event in events] == ["success", "success", "success", "success"]
    memories = store.list_active_memories("u001")
    assert any("gravity still needs practice" in memory.content for memory in memories)
