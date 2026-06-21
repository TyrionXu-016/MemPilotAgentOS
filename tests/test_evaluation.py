from agentos.evaluation import evaluate_planners, seed_learning_memories


def test_layered_memory_beats_no_memory_on_preference_match(store):
    seed_learning_memories(store)
    result = evaluate_planners(store)
    no_memory = result["no_memory"]
    layered = result["layered_memory"]
    assert layered.preference_match_rate > no_memory.preference_match_rate
    assert layered.executable_plan_rate == 1.0
