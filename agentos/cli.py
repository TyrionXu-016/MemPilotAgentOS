from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from agentos.evaluation import evaluate_planners, export_evaluation_artifacts, seed_learning_memories
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
    store = open_store(db)
    results = evaluate_planners(store)
    print({name: result.model_dump() for name, result in results.items()})


@app.command()
def export_results(
    db: Path = Path("data/agentos.sqlite"),
    output_dir: Path = Path("docs/results"),
    stem: str = "agentos-evaluation",
    seed_demo: bool = True,
) -> None:
    store = open_store(db)
    if seed_demo:
        seed_learning_memories(store)
    paths = export_evaluation_artifacts(store=store, output_dir=output_dir, stem=stem)
    print({name: str(path) for name, path in paths.items()})


if __name__ == "__main__":
    app()
