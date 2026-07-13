from __future__ import annotations

from agentos.deepseek import DeepSeekPlanGenerator
from agentos.models import MemoryEvidence, Plan, PlanContext, PlanStep
from agentos.retriever import MemoryRetriever
from agentos.skills import SkillRegistry


class DeepSeekPlanner:
    def __init__(
        self,
        registry: SkillRegistry,
        generator: DeepSeekPlanGenerator,
        planner_name: str,
        retriever: MemoryRetriever | None = None,
    ):
        self.registry = registry
        self.generator = generator
        self.planner_name = planner_name
        self.retriever = retriever

    def plan(self, user_id: str, goal: str) -> Plan:
        retrieved = self.retriever.retrieve(user_id, goal, limit=5) if self.retriever else []
        evidence = [
            MemoryEvidence(
                memory_id=item.memory.id,
                memory_type=item.memory.memory_type,
                content=item.memory.content,
                score=item.score,
            )
            for item in retrieved
        ]
        generation = self.generator.generate(
            user_id,
            goal,
            [spec.model_dump() for spec in self.registry.specs.values()],
            [item.model_dump() for item in evidence],
        )
        draft = generation.draft
        available_memory_ids = {
            item.memory_id for item in evidence if item.memory_id is not None
        }
        unavailable_memory_ids = set(draft.memory_ids) - available_memory_ids
        if unavailable_memory_ids:
            raise ValueError(
                f"DeepSeek referenced unavailable memory IDs: {sorted(unavailable_memory_ids)}"
            )
        plan = Plan(
            goal=goal,
            basis=[item.content for item in evidence] or ["仅使用当前用户目标，不读取长期记忆"],
            planner_name=self.planner_name,
            context=PlanContext(
                scenario_group=draft.scenario_group,
                interaction_style=draft.interaction_style,
                task_items=draft.task_items,
            ),
            evidence=evidence,
            used_memory_ids=draft.memory_ids,
            steps=[PlanStep(skill=step.skill, params=step.params) for step in draft.steps],
            generation_metadata=generation.model_dump(exclude={"draft"}),
        )
        self.registry.validate_plan(plan)
        return plan
