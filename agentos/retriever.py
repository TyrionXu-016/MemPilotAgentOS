from agentos.models import RetrievedMemory
from agentos.storage import AgentOSStore


class MemoryRetriever:
    def __init__(self, store: AgentOSStore):
        self.store = store

    def retrieve(self, user_id: str, goal: str, limit: int = 5) -> list[RetrievedMemory]:
        goal_lower = goal.lower()
        records = self.store.list_active_memories(user_id)
        scored: list[RetrievedMemory] = []
        for record in records:
            haystack = " ".join([record.content, *record.tags]).lower()
            lexical = self._lexical_score(goal_lower, haystack)
            type_bonus = 0.4 if record.memory_type in {"preference", "feedback", "task"} else 0.0
            score = lexical + type_bonus + record.importance + record.confidence * 0.2
            if score >= 0.85:
                scored.append(
                    RetrievedMemory(
                        memory=record,
                        score=score,
                        reason="keyword+memory_type+importance+confidence",
                    )
                )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def _lexical_score(self, goal: str, haystack: str) -> float:
        score = 0.0
        keyword_pairs = {
            "太空": ["太空", "space"],
            "英语": ["英语", "english"],
            "单词": ["单词", "word", "vocabulary"],
            "复习": ["复习", "review", "练习"],
            "gravity": ["gravity"],
            "家庭": ["家庭", "family"],
            "作业": ["作业", "homework"],
            "数学": ["数学", "math"],
            "分数": ["分数", "fraction"],
            "故事": ["故事", "story"],
            "鼓励": ["鼓励", "encouragement"],
            "睡前": ["睡前", "bedtime"],
            "饮水": ["饮水", "温水", "warm_water"],
            "温水": ["温水", "warm_water"],
            "水杯": ["水杯", "cup"],
            "书桌": ["书桌", "desk"],
            "晚间": ["晚间", "evening"],
            "生活": ["生活", "home"],
        }
        for goal_key, memory_keys in keyword_pairs.items():
            if goal_key in goal and any(memory_key in haystack for memory_key in memory_keys):
                score += 1.0
        for token in goal.split():
            if token and token in haystack:
                score += 0.5
        return score
