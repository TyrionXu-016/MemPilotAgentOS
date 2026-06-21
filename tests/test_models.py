from agentos.models import MemoryRecord, Plan, PlanStep, SkillSpec


def test_memory_record_defaults_are_stable():
    record = MemoryRecord(
        user_id="u001",
        memory_type="preference",
        content="用户喜欢太空主题互动",
        source="interaction",
    )
    assert record.importance == 0.5
    assert record.confidence == 0.8
    assert record.is_active is True


def test_plan_references_executable_skills():
    plan = Plan(
        goal="复习英语单词",
        basis=["用户喜欢太空主题", "用户上次 gravity 答错"],
        steps=[
            PlanStep(skill="generate_quiz", params={"theme": "space", "words": ["gravity"]}),
            PlanStep(skill="ask_question", params={"mode": "interactive"}),
        ],
    )
    assert plan.steps[0].skill == "generate_quiz"
    assert plan.basis[1].startswith("用户上次")


def test_skill_spec_requires_declared_params():
    spec = SkillSpec(
        name="generate_quiz",
        description="Generate a themed vocabulary quiz",
        required_params=["theme", "words"],
        output_type="quiz",
    )
    assert spec.required_params == ["theme", "words"]
