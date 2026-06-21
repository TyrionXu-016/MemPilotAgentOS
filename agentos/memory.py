from agentos.models import MemoryRecord
from agentos.storage import AgentOSStore


class MemoryManager:
    def __init__(self, store: AgentOSStore):
        self.store = store

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        return self.store.add_memory(record)

    def record_feedback(self, user_id: str, goal: str, feedback: str) -> MemoryRecord:
        return self.store.add_memory(
            MemoryRecord(
                user_id=user_id,
                memory_type="feedback",
                content=f"任务目标：{goal}；执行反馈：{feedback}",
                source="runtime_feedback",
                importance=0.7,
                confidence=0.9,
                tags=["feedback", goal],
            )
        )
