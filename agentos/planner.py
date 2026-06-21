from __future__ import annotations

from agentos.models import MemoryRecord, MemoryType, Plan, PlanStep
from agentos.retriever import MemoryRetriever
from agentos.skills import SkillRegistry


class NoMemoryPlanner:
    planner_name = "no_memory"

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def plan(self, user_id: str, goal: str) -> Plan:
        words = ["review"]
        plan = Plan(
            goal=goal,
            basis=["仅使用当前用户目标，不读取长期记忆"],
            planner_name=self.planner_name,
            steps=[
                PlanStep(skill="retrieve_learning_history", params={"user_id": user_id}),
                PlanStep(skill="generate_quiz", params={"theme": "neutral", "words": words}),
                PlanStep(skill="ask_question", params={"mode": "interactive"}),
                PlanStep(skill="evaluate_answer", params={"expected_words": words}),
                PlanStep(skill="update_memory", params={"feedback": "completed generic review"}),
            ],
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
        retrieved = self.retriever.retrieve(user_id=user_id, goal=goal, limit=5)
        memories = self._filter_memories([item.memory for item in retrieved])
        basis = [memory.content for memory in memories] or ["未命中长期记忆，退化为当前目标规划"]
        theme = self._select_theme(memories)
        words = self._select_practice_items(memories)
        plan = Plan(
            goal=goal,
            basis=basis,
            planner_name=self.planner_name,
            steps=[
                PlanStep(skill="retrieve_learning_history", params={"user_id": user_id}),
                PlanStep(skill="generate_quiz", params={"theme": theme, "words": words}),
                PlanStep(skill="ask_question", params={"mode": "interactive"}),
                PlanStep(skill="evaluate_answer", params={"expected_words": words}),
                PlanStep(
                    skill="update_memory",
                    params={"feedback": f"reviewed {', '.join(words)} with theme {theme}"},
                ),
            ],
        )
        self.registry.validate_plan(plan)
        return plan

    def _filter_memories(self, memories: list[MemoryRecord]) -> list[MemoryRecord]:
        if self.allowed_memory_types is None:
            return memories
        return [memory for memory in memories if memory.memory_type in self.allowed_memory_types]

    def _has_space_preference(self, memories: list[MemoryRecord]) -> bool:
        return any(
            memory.memory_type in {"preference", "profile"}
            and ("太空" in memory.content or "space" in memory.tags)
            for memory in memories
        )

    def _has_gravity_feedback(self, memories: list[MemoryRecord]) -> bool:
        return any(
            memory.memory_type in {"feedback", "task"}
            and ("gravity" in memory.content.lower() or "gravity" in memory.tags)
            for memory in memories
        )

    def _has_story_preference(self, memories: list[MemoryRecord]) -> bool:
        return any(
            memory.memory_type in {"preference", "profile"}
            and ("故事" in memory.content or "story" in memory.tags or "encouragement" in memory.tags)
            for memory in memories
        )

    def _has_fraction_feedback(self, memories: list[MemoryRecord]) -> bool:
        return any(
            memory.memory_type in {"feedback", "task"}
            and ("分数" in memory.content or "fraction" in memory.tags)
            for memory in memories
        )

    def _has_home_preference(self, memories: list[MemoryRecord]) -> bool:
        return any(
            memory.memory_type in {"preference", "profile", "scene"}
            and (
                "温水" in memory.content
                or "书桌" in memory.content
                or "home" in memory.tags
                or "warm_water" in memory.tags
            )
            for memory in memories
        )

    def _has_warm_water_feedback(self, memories: list[MemoryRecord]) -> bool:
        return any(
            memory.memory_type in {"feedback", "task", "scene"}
            and (
                "温水" in memory.content
                or "睡前" in memory.content
                or "水杯" in memory.content
                or "warm_water" in memory.tags
            )
            for memory in memories
        )

    def _select_theme(self, memories: list[MemoryRecord]) -> str:
        if self._has_space_preference(memories):
            return "space"
        if self._has_story_preference(memories):
            return "story"
        if self._has_home_preference(memories):
            return "home"
        return "neutral"

    def _select_practice_items(self, memories: list[MemoryRecord]) -> list[str]:
        if self._has_gravity_feedback(memories):
            return ["gravity"]
        if self._has_fraction_feedback(memories):
            return ["fraction"]
        if self._has_warm_water_feedback(memories):
            return ["warm_water"]
        return ["review"]


class MemoryFilteredPlanner(LayeredMemoryPlanner):
    def __init__(
        self,
        registry: SkillRegistry,
        retriever: MemoryRetriever,
        planner_name: str,
        allowed_memory_types: set[MemoryType],
    ):
        super().__init__(
            registry=registry,
            retriever=retriever,
            allowed_memory_types=allowed_memory_types,
            planner_name=planner_name,
        )
