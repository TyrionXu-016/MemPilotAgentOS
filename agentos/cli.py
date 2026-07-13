from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from agentos.benchmark.dataset import DEFAULT_DATASET, load_benchmark
from agentos.benchmark.export import export_benchmark_artifacts
from agentos.benchmark.runner import run_deterministic_suite
from agentos.evaluation import seed_learning_memories
from agentos.memory import MemoryManager
from agentos.planner import LayeredMemoryPlanner
from agentos.retriever import MemoryRetriever
from agentos.runtime import AgentOSRuntime
from agentos.skills import build_default_registry
from agentos.storage import AgentOSStore

app = typer.Typer(help="Embodied AgentOS prototype CLI")


def open_store(db: Path) -> AgentOSStore:
    store = AgentOSStore(db)
    store.migrate()
    return store


@app.command()
def seed(db: Path = Path("data/agentos.sqlite")) -> None:
    store = open_store(db)
    seed_learning_memories(store)
    print("[green]seeded learning-companion memories[/green]")


@app.command()
def run(goal: str, user_id: str = "u001", db: Path = Path("data/agentos.sqlite")) -> None:
    store = open_store(db)
    registry = build_default_registry()
    planner = LayeredMemoryPlanner(registry, MemoryRetriever(store))
    plan = planner.plan(user_id=user_id, goal=goal)
    events = AgentOSRuntime(registry, MemoryManager(store)).run(user_id=user_id, plan=plan)
    print(
        {
            "plan": plan.model_dump(),
            "events": [event.model_dump() for event in events],
        }
    )


@app.command()
def evaluate(db: Path = Path("data/agentos.sqlite")) -> None:
    _ = db
    result = run_deterministic_suite(load_benchmark())
    summaries = [
        summary.model_dump()
        for summary in result.statistics
        if summary.condition == "clean" and summary.scenario_group == "all"
    ]
    print(summaries)


@app.command()
def export_results(
    db: Path = Path("data/agentos.sqlite"),
    dataset: Path = DEFAULT_DATASET,
    output_dir: Path = Path("docs/results"),
    stem: str = "agentos-evaluation",
    seed_demo: bool = True,
    include_deepseek: bool = False,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_repeats: int = 5,
    concurrency: int = 5,
    cache: Path = Path(".agentos-cache/deepseek-runs.jsonl"),
) -> None:
    _ = db, seed_demo
    paths = export_benchmark_artifacts(
        benchmark=load_benchmark(dataset),
        output_dir=output_dir,
        stem=stem,
        include_deepseek=include_deepseek,
        deepseek_model=deepseek_model,
        deepseek_repeats=deepseek_repeats,
        concurrency=concurrency,
        cache_path=cache if include_deepseek else None,
        dataset_path=dataset,
    )
    print({name: str(path) for name, path in paths.items()})


if __name__ == "__main__":
    app()
