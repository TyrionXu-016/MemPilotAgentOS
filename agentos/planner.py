from agentos.models import Plan, PlanStep
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

    def __init__(self, registry: SkillRegistry, retriever: MemoryRetriever):
        self.registry = registry
        self.retriever = retriever

    def plan(self, user_id: str, goal: str) -> Plan:
        retrieved = self.retriever.retrieve(user_id=user_id, goal=goal, limit=5)
        memories = [item.memory for item in retrieved]
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

    def _has_space_preference(self, memories: list) -> bool:
        return any("太空" in memory.content or "space" in memory.tags for memory in memories)

    def _has_gravity_feedback(self, memories: list) -> bool:
        return any("gravity" in memory.content.lower() or "gravity" in memory.tags for memory in memories)

    def _has_story_preference(self, memories: list) -> bool:
        return any(
            "故事" in memory.content or "story" in memory.tags or "encouragement" in memory.tags
            for memory in memories
        )

    def _has_fraction_feedback(self, memories: list) -> bool:
        return any("分数" in memory.content or "fraction" in memory.tags for memory in memories)

    def _select_theme(self, memories: list) -> str:
        if self._has_space_preference(memories):
            return "space"
        if self._has_story_preference(memories):
            return "story"
        return "neutral"

    def _select_practice_items(self, memories: list) -> list[str]:
        if self._has_gravity_feedback(memories):
            return ["gravity"]
        if self._has_fraction_feedback(memories):
            return ["fraction"]
        return ["review"]
