from collections import Counter
from pathlib import Path

from agentos.benchmark.dataset import load_benchmark
from agentos.benchmark.export import benchmark_sha256


DATASET = Path("experiments/data/agentos-benchmark-v1.json")


def test_benchmark_v1_has_balanced_72_cases():
    benchmark = load_benchmark(DATASET)

    assert benchmark.version == "1.0"
    assert len(benchmark.cases) == 72
    assert len({case.case_id for case in benchmark.cases}) == 72
    assert Counter(case.scenario_group for case in benchmark.cases) == {
        "learning": 24,
        "family_education": 24,
        "home_service": 24,
    }
    for group in {case.scenario_group for case in benchmark.cases}:
        explicitness = Counter(
            case.explicitness for case in benchmark.cases if case.scenario_group == group
        )
        assert explicitness == {
            "full": 6,
            "style_only": 6,
            "item_only": 6,
            "implicit": 6,
        }


def test_benchmark_references_only_declared_memories():
    benchmark = load_benchmark(DATASET)

    for case in benchmark.cases:
        keys = {memory.memory_key for memory in case.memories}
        assert set(case.relevant_memory_keys) <= keys
        assert case.expected_items
        assert case.required_skills


def test_benchmark_hash_is_stable_across_loads():
    assert benchmark_sha256(load_benchmark(DATASET)) == benchmark_sha256(load_benchmark(DATASET))
