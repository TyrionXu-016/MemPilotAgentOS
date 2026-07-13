from __future__ import annotations

from agentos.models import MemoryEvidence, MemoryType, Plan, PlanContext, PlanStep, RetrievedMemory
from agentos.ontology import Selection, select_from_goal, select_from_memories
from agentos.retriever import MemoryRetriever
from agentos.skills import SkillRegistry


class NoMemoryPlanner:
    planner_name = "no_memory"

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def plan(self, user_id: str, goal: str) -> Plan:
        selected = select_from_goal(goal)
        plan = _build_plan(
            user_id=user_id,
            goal=goal,
            planner_name=self.planner_name,
            selected=selected,
            basis=["仅使用当前用户目标，不读取长期记忆"],
        )
        self.registry.validate_plan(plan)
        return plan


class LayeredMemoryPlanner:
    planner_name = "layered_memory"
    allowed_memory_types: set[MemoryType] | None = None

    def __init__(
        self,
        registry: SkillRegistry,
        retriever: MemoryRetriever,
        allowed_memory_types: set[MemoryType] | None = None,
        planner_name: str | None = None,
    ):
        self.registry = registry
        self.retriever = retriever
        self.allowed_memory_types = allowed_memory_types
        if planner_name is not None:
            self.planner_name = planner_name

    def plan(self, user_id: str, goal: str) -> Plan:
        retrieved = self._filter_retrieved(self.retriever.retrieve(user_id=user_id, goal=goal, limit=5))
        memories = [item.memory for item in retrieved]
        goal_selection = select_from_goal(goal)
        memory_selection = select_from_memories(memories)
        selected = Selection(
            scenario_group=memory_selection.scenario_group or goal_selection.scenario_group,
            style=goal_selection.style or memory_selection.style,
            item=goal_selection.item or memory_selection.item,
            used_memory_ids=memory_selection.used_memory_ids,
        )
        basis = [memory.content for memory in memories] or ["未命中长期记忆，退化为当前目标规划"]
        evidence = [
            MemoryEvidence(
                memory_id=item.memory.id,
                memory_type=item.memory.memory_type,
                content=item.memory.content,
                score=item.score,
            )
            for item in retrieved
        ]
        plan = _build_plan(
            user_id=user_id,
            goal=goal,
            planner_name=self.planner_name,
            selected=selected,
            basis=basis,
            evidence=evidence,
        )
        self.registry.validate_plan(plan)
        return plan

    def _filter_retrieved(self, retrieved: list[RetrievedMemory]) -> list[RetrievedMemory]:
        if self.allowed_memory_types is None:
            return retrieved
        return [item for item in retrieved if item.memory.memory_type in self.allowed_memory_types]


class MemoryFilteredPlanner(LayeredMemoryPlanner):
    def __init__(
        self,
        registry: SkillRegistry,
        retriever: MemoryRetriever,
        planner_name: str,
        allowed_memory_types: set[MemoryType],
    ):
        super().__init__(registry, retriever, allowed_memory_types, planner_name)


def _build_plan(
    user_id: str,
    goal: str,
    planner_name: str,
    selected: Selection,
    basis: list[str],
    evidence: list[MemoryEvidence] | None = None,
) -> Plan:
    group = selected.scenario_group or "learning"
    style = selected.style or "neutral"
    items = [selected.item or "review"]
    if group == "home_service":
        steps = [
            PlanStep(skill="retrieve_home_context", params={"user_id": user_id}),
            PlanStep(skill="prepare_home_assistance", params={"theme": style, "words": items}),
            PlanStep(skill="notify_user", params={"mode": "interactive"}),
            PlanStep(skill="verify_home_task", params={"expected_words": items}),
            PlanStep(skill="update_memory", params={"feedback": f"completed {', '.join(items)} with {style}"}),
        ]
    else:
        steps = [
            PlanStep(skill="retrieve_learning_history", params={"user_id": user_id}),
            PlanStep(skill="generate_quiz", params={"theme": style, "words": items}),
            PlanStep(skill="ask_question", params={"mode": "interactive"}),
            PlanStep(skill="evaluate_answer", params={"expected_words": items}),
            PlanStep(skill="update_memory", params={"feedback": f"reviewed {', '.join(items)} with theme {style}"}),
        ]
    return Plan(
        goal=goal,
        basis=basis,
        steps=steps,
        planner_name=planner_name,
        context=PlanContext(
            scenario_group=group,
            interaction_style=style,
            task_items=items,
        ),
        evidence=evidence or [],
        used_memory_ids=list(selected.used_memory_ids),
    )
