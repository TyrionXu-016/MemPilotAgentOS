from agentos.memory import MemoryManager
from agentos.models import MemoryRecord


def test_store_round_trips_memory(store):
    record = MemoryRecord(
        user_id="u001",
        memory_type="preference",
        content="用户喜欢太空主题互动",
        source="seed",
        tags=["learning", "theme"],
    )
    saved = store.add_memory(record)
    loaded = store.get_memory(saved.id)
    assert loaded is not None
    assert loaded.content == "用户喜欢太空主题互动"
    assert loaded.tags == ["learning", "theme"]


def test_memory_manager_records_feedback_as_memory(store):
    manager = MemoryManager(store)
    saved = manager.record_feedback(
        user_id="u001",
        goal="复习英语单词",
        feedback="gravity 答错，orbit 答对",
    )
    assert saved.memory_type == "feedback"
    assert "gravity" in saved.content
    assert saved.importance == 0.7
