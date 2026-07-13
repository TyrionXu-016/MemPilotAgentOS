from collections.abc import Callable
from typing import Any

from agentos.models import Plan, SkillSpec


SkillHandler = Callable[[dict[str, Any]], dict[str, Any]]


class SkillRegistry:
    def __init__(self) -> None:
        self.specs: dict[str, SkillSpec] = {}
        self.handlers: dict[str, SkillHandler] = {}

    def register(self, spec: SkillSpec, handler: SkillHandler) -> None:
        self.specs[spec.name] = spec
        self.handlers[spec.name] = handler

    def validate_plan(self, plan: Plan) -> None:
        for step in plan.steps:
            if step.skill not in self.specs:
                raise ValueError(f"Unknown skill: {step.skill}")
            spec = self.specs[step.skill]
            missing = [name for name in spec.required_params if name not in step.params]
            if missing:
                raise ValueError(f"Skill {step.skill} missing params: {', '.join(missing)}")

    def execute(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if skill_name not in self.specs:
            raise ValueError(f"Unknown skill: {skill_name}")
        spec = self.specs[skill_name]
        missing = [name for name in spec.required_params if name not in params]
        if missing:
            raise ValueError(f"Skill {skill_name} missing params: {', '.join(missing)}")
        return self.handlers[skill_name](params)


def build_default_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="retrieve_home_context",
            description="Read home-service scene and task context",
            required_params=["user_id"],
            output_type="home_context",
        ),
        lambda params: {"user_id": params["user_id"], "context_loaded": True},
    )
    registry.register(
        SkillSpec(
            name="prepare_home_assistance",
            description="Prepare a home assistance task",
            required_params=["theme", "words"],
            output_type="home_task",
        ),
        lambda params: {"style": params["theme"], "tasks": params["words"]},
    )
    registry.register(
        SkillSpec(
            name="notify_user",
            description="Notify the user through the simulated terminal",
            required_params=["mode"],
            output_type="notification",
        ),
        lambda params: {"mode": params["mode"], "notified": True},
    )
    registry.register(
        SkillSpec(
            name="verify_home_task",
            description="Verify expected home assistance tasks",
            required_params=["expected_words"],
            output_type="verification",
        ),
        lambda params: {"completed": True, "tasks": params["expected_words"]},
    )
    registry.register(
        SkillSpec(
            name="retrieve_learning_history",
            description="Read recent learning history",
            required_params=["user_id"],
            output_type="history",
        ),
        lambda params: {"user_id": params["user_id"], "recent_wrong_words": ["gravity"]},
    )
    registry.register(
        SkillSpec(
            name="generate_quiz",
            description="Generate a themed vocabulary quiz",
            required_params=["theme", "words"],
            output_type="quiz",
        ),
        lambda params: {
            "theme": params["theme"],
            "questions": [
                {
                    "word": word,
                    "prompt": f"Explain {word} in a {params['theme']} themed interaction.",
                }
                for word in params["words"]
            ],
        },
    )
    registry.register(
        SkillSpec(
            name="ask_question",
            description="Ask one question through simulated embodied terminal",
            required_params=["mode"],
            output_type="utterance",
        ),
        lambda params: {"mode": params["mode"], "asked": True},
    )
    registry.register(
        SkillSpec(
            name="evaluate_answer",
            description="Evaluate learner answer",
            required_params=["expected_words"],
            output_type="score",
        ),
        lambda params: {"score": 0.8, "wrong_words": params["expected_words"][:1]},
    )
    registry.register(
        SkillSpec(
            name="update_memory",
            description="Persist task feedback",
            required_params=["feedback"],
            output_type="memory_update",
        ),
        lambda params: {"feedback": params["feedback"], "stored": True},
    )
    return registry
