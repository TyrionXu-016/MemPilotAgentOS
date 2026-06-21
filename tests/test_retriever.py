from agentos.models import MemoryRecord
from agentos.retriever import MemoryRetriever


def test_retriever_prioritizes_relevant_preferences(store):
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="preference",
            content="用户喜欢太空主题互动",
            source="seed",
            importance=0.9,
            tags=["theme", "space"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="preference",
            content="用户不喜欢恐龙主题互动",
            source="seed",
            importance=0.6,
            tags=["theme", "dinosaur"],
        )
    )
    retriever = MemoryRetriever(store)
    results = retriever.retrieve(user_id="u001", goal="用太空主题复习英语", limit=2)
    assert results[0].memory.content == "用户喜欢太空主题互动"
    assert results[0].score > results[1].score


def test_retriever_filters_other_users(store):
    store.add_memory(
        MemoryRecord(
            user_id="u002",
            memory_type="preference",
            content="用户喜欢太空主题互动",
            source="seed",
            importance=1.0,
            tags=["space"],
        )
    )
    retriever = MemoryRetriever(store)
    assert retriever.retrieve(user_id="u001", goal="太空主题", limit=3) == []
