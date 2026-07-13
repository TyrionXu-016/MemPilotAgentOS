from agentos.evaluation import (
    evaluate_experiment_suite,
    evaluate_planners,
    export_evaluation_artifacts,
    seed_learning_memories,
)


def test_layered_memory_beats_no_memory_on_scaled_benchmark(store):
    seed_learning_memories(store)
    result = evaluate_planners(store)

    assert set(result) == {"no_memory", "preference_only", "feedback_only", "layered_memory"}
    assert result["no_memory"].scenario_count == 72
    assert result["layered_memory"].preference_match_rate == 1.0
    assert result["layered_memory"].preference_match_rate > result["no_memory"].preference_match_rate


def test_scaled_ablation_separates_preference_and_feedback_effects(store):
    result = evaluate_planners(store)

    assert result["no_memory"].preference_match_rate == 0.5
    assert result["no_memory"].task_completion_rate == 0.5
    assert result["preference_only"].preference_match_rate == 1.0
    assert result["preference_only"].task_completion_rate == 0.5
    assert result["feedback_only"].preference_match_rate == 0.5
    assert result["feedback_only"].task_completion_rate == 1.0
    assert result["layered_memory"].preference_match_rate == 1.0
    assert result["layered_memory"].task_completion_rate == 1.0


def test_experiment_suite_includes_scaled_robustness_and_feedback(store):
    suite = evaluate_experiment_suite(store)

    assert len(suite.clean_runs) == 288
    assert len(suite.robustness_runs) == 288
    assert {run.condition for run in suite.robustness_runs} == {
        "clean",
        "irrelevant_noise",
        "conflict_low_confidence",
        "stale_low_importance",
    }
    assert len(suite.feedback_runs) == 36


def test_exports_scaled_paper_ready_result_artifacts(store, tmp_path):
    paths = export_evaluation_artifacts(store, output_dir=tmp_path, stem="agentos-evaluation")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    csv = paths["csv"].read_text(encoding="utf-8")
    cases = paths["cases_csv"].read_text(encoding="utf-8").splitlines()

    assert "AgentOS 规模化记忆增强规划实验结果" in markdown
    assert "规模化场景泛化实验" in markdown
    assert "统计显著性" in markdown
    assert "planner_name,sample_count" in csv
    assert len(cases) == 1 + 288 + 288 + 36
    assert paths["confidence_svg"].exists()
    assert paths["deepseek_svg"].exists()
