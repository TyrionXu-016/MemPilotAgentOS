from collections import defaultdict
import json

from agentos.benchmark.dataset import load_benchmark
from agentos.benchmark.models import BenchmarkSuite
from agentos.benchmark.runner import run_deepseek_suite, run_deterministic_suite
from agentos.deepseek import DeepSeekPlanGenerator, DeepSeekRawResponse


def test_deterministic_suite_has_expected_counts_and_rates():
    result = run_deterministic_suite(load_benchmark())

    assert len(result.clean_runs) == 288
    assert len(result.robustness_runs) == 288
    assert len(result.feedback_runs) == 36
    assert {run.experiment for run in result.clean_runs} == {"clean"}
    assert {run.experiment for run in result.robustness_runs} == {"robustness"}
    assert {run.experiment for run in result.feedback_runs} == {"feedback_loop"}

    rates = defaultdict(dict)
    for summary in result.statistics:
        if summary.condition == "clean" and summary.scenario_group == "all":
            rates[summary.planner_name][summary.metric] = summary.value
            assert summary.sample_count == 72

    assert rates["no_memory"]["preference_match_rate"] == 0.5
    assert rates["no_memory"]["task_coverage_rate"] == 0.5
    assert rates["preference_only"]["preference_match_rate"] == 1.0
    assert rates["preference_only"]["task_coverage_rate"] == 0.5
    assert rates["feedback_only"]["preference_match_rate"] == 0.5
    assert rates["feedback_only"]["task_coverage_rate"] == 1.0
    assert rates["layered_memory"]["preference_match_rate"] == 1.0
    assert rates["layered_memory"]["task_coverage_rate"] == 1.0
    assert rates["layered_memory"]["latency_ms_mean"] >= 0
    assert rates["layered_memory"]["latency_ms_p95"] >= rates["layered_memory"]["latency_ms_p50"]


def test_feedback_writeback_improves_all_implicit_cases():
    result = run_deterministic_suite(load_benchmark())
    before = [run for run in result.feedback_runs if run.condition == "before_feedback_writeback"]
    after = [run for run in result.feedback_runs if run.condition == "after_feedback_writeback"]

    assert len(before) == len(after) == 18
    assert sum(run.task_coverage for run in before) == 0
    assert sum(run.task_coverage for run in after) == 18


def test_deepseek_runs_resume_from_jsonl_cache(tmp_path):
    benchmark = load_benchmark()
    one_case = BenchmarkSuite(version=benchmark.version, cases=[benchmark.cases[0]])
    calls = []

    def request(payload):
        calls.append(payload)
        content = json.dumps(
            {
                "scenario_group": "learning",
                "interaction_style": "space",
                "task_items": ["gravity"],
                "memory_ids": [],
                "steps": [
                    {"skill": "retrieve_learning_history", "params": {"user_id": "lear01"}},
                    {"skill": "generate_quiz", "params": {"theme": "space", "words": ["gravity"]}},
                    {"skill": "ask_question", "params": {"mode": "interactive"}},
                    {"skill": "evaluate_answer", "params": {"expected_words": ["gravity"]}},
                    {"skill": "update_memory", "params": {"feedback": "reviewed gravity"}},
                ],
            }
        )
        return DeepSeekRawResponse(
            content=content,
            response_id=f"response-{len(calls)}",
            model="deepseek-v4-flash",
            system_fingerprint="fp-test",
        )

    def factory():
        return DeepSeekPlanGenerator(request=request)

    cache = tmp_path / "runs.jsonl"
    first = run_deepseek_suite(
        one_case,
        repeats=2,
        concurrency=2,
        cache_path=cache,
        generator_factory=factory,
    )
    second = run_deepseek_suite(
        one_case,
        repeats=2,
        concurrency=2,
        cache_path=cache,
        generator_factory=factory,
    )

    assert len(first) == len(second) == 4
    assert len(calls) == 4
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 4
