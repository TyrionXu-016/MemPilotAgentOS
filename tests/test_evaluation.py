from agentos.evaluation import (
    evaluate_experiment_suite,
    export_evaluation_artifacts,
    evaluate_planners,
    seed_learning_memories,
)


def test_layered_memory_beats_no_memory_on_preference_match(store):
    seed_learning_memories(store)
    result = evaluate_planners(store)
    no_memory = result["no_memory"]
    layered = result["layered_memory"]
    assert set(result) == {"no_memory", "preference_only", "feedback_only", "layered_memory"}
    assert layered.preference_match_rate > no_memory.preference_match_rate
    assert layered.executable_plan_rate == 1.0


def test_ablation_planners_separate_preference_and_feedback_effects(store):
    seed_learning_memories(store)
    result = evaluate_planners(store)
    preference_only = result["preference_only"]
    feedback_only = result["feedback_only"]
    layered = result["layered_memory"]
    assert preference_only.preference_match_rate == 1.0
    assert preference_only.task_completion_rate == 0.0
    assert feedback_only.preference_match_rate == 0.0
    assert feedback_only.task_completion_rate == 1.0
    assert layered.preference_match_rate == 1.0
    assert layered.task_completion_rate == 1.0


def test_experiment_suite_includes_robustness_variants(store):
    seed_learning_memories(store)
    suite = evaluate_experiment_suite(store)
    assert [row.planner_name for row in suite.robustness.rows] == [
        "clean",
        "irrelevant_noise",
        "conflict_low_confidence",
        "stale_low_importance",
    ]
    assert all(row.preference_match_rate == 1.0 for row in suite.robustness.rows)
    assert all(row.task_completion_rate == 1.0 for row in suite.robustness.rows)


def test_exports_paper_ready_result_artifacts(store, tmp_path):
    seed_learning_memories(store)
    paths = export_evaluation_artifacts(store, output_dir=tmp_path, stem="agentos-evaluation")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    csv = paths["csv"].read_text(encoding="utf-8")
    scenario_csv = paths["scenario_csv"].read_text(encoding="utf-8")
    robustness_csv = paths["robustness_csv"].read_text(encoding="utf-8")
    svg = paths["svg"].read_text(encoding="utf-8")
    assert "AgentOS 记忆增强规划实验结果" in markdown
    assert "场景泛化实验" in markdown
    assert "消融实验" in markdown
    assert "鲁棒性实验" in markdown
    assert "| layered_memory | 9 | 1.00 | 1.00 | 1.00 | 1.00 |" in markdown
    assert "planner_name,scenario_count,memory_hit_rate" in csv
    assert "preference_only,9,1.0,1.0,1.0,0.0" in csv
    assert "feedback_only,9,1.0,0.0,1.0,1.0" in csv
    assert "layered_memory,9,1.0,1.0,1.0,1.0" in csv
    assert "home_service:layered_memory,3,1.0,1.0,1.0,1.0" in scenario_csv
    assert "conflict_low_confidence,9,1.0,1.0,1.0,1.0" in robustness_csv
    assert "<svg" in svg
    assert "AgentOS 总体规划指标" in svg
    assert paths["overall_svg"].exists()
    assert paths["ablation_svg"].exists()
    assert paths["robustness_svg"].exists()
