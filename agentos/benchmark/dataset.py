from __future__ import annotations

from pathlib import Path

from agentos.benchmark.models import BenchmarkSuite


DEFAULT_DATASET = Path("experiments/data/agentos-benchmark-v1.json")


def load_benchmark(path: Path | str = DEFAULT_DATASET) -> BenchmarkSuite:
    return BenchmarkSuite.model_validate_json(Path(path).read_text(encoding="utf-8"))
