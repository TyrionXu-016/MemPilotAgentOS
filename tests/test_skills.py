import pytest

from agentos.models import Plan, PlanStep
from agentos.skills import build_default_registry


def test_registry_rejects_missing_required_param():
    registry = build_default_registry()
    plan = Plan(goal="复习", basis=[], steps=[PlanStep(skill="generate_quiz", params={"theme": "space"})])
    with pytest.raises(ValueError, match="words"):
        registry.validate_plan(plan)


def test_registry_executes_generate_quiz():
    registry = build_default_registry()
    output = registry.execute("generate_quiz", {"theme": "space", "words": ["gravity"]})
    assert output["questions"][0]["word"] == "gravity"
    assert output["theme"] == "space"
