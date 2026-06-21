from agentos.evaluation import export_evaluation_artifacts, evaluate_planners, seed_learning_memories


def test_layered_memory_beats_no_memory_on_preference_match(store):
    seed_learning_memories(store)
    result = evaluate_planners(store)
    no_memory = result["no_memory"]
    layered = result["layered_memory"]
    assert layered.preference_match_rate > no_memory.preference_match_rate
    assert layered.executable_plan_rate == 1.0


def test_exports_paper_ready_result_artifacts(store, tmp_path):
    seed_learning_memories(store)
    paths = export_evaluation_artifacts(store, output_dir=tmp_path, stem="agentos-evaluation")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    csv = paths["csv"].read_text(encoding="utf-8")
    svg = paths["svg"].read_text(encoding="utf-8")
    assert "AgentOS 记忆增强规划实验结果" in markdown
    assert "| layered_memory | 6 | 1.00 | 1.00 | 1.00 | 1.00 |" in markdown
    assert "planner_name,scenario_count,memory_hit_rate" in csv
    assert "layered_memory,6,1.0,1.0,1.0,1.0" in csv
    assert "<svg" in svg
    assert "AgentOS 记忆增强规划实验指标" in svg
