"""Compatibility facade for the scaled AgentOS benchmark."""

from __future__ import annotations

from pathlib import Path

from agentos.benchmark.dataset import load_benchmark
from agentos.benchmark.export import export_benchmark_artifacts
from agentos.benchmark.runner import run_deterministic_suite
from agentos.models import EvaluationResult, MemoryRecord
from agentos.storage import AgentOSStore


def seed_learning_memories(store: AgentOSStore) -> None:
    seed_learning_companion_memories(store)
    seed_family_education_memories(store)
    seed_home_service_memories(store)


def seed_learning_companion_memories(store: AgentOSStore) -> None:
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="preference",
            content="用户喜欢太空主题互动",
            source="seed",
            importance=0.9,
            confidence=0.95,
            tags=["space", "theme", "learning"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="u001",
            memory_type="feedback",
            content="任务目标：英语复习；执行反馈：gravity 答错，需要继续复习",
            source="seed",
            importance=0.8,
            confidence=0.9,
            tags=["gravity", "english", "feedback"],
        )
    )


def seed_family_education_memories(store: AgentOSStore) -> None:
    store.add_memory(
        MemoryRecord(
            user_id="family001",
            memory_type="preference",
            content="孩子更容易接受故事化讲解和鼓励式反馈",
            source="seed",
            importance=0.9,
            confidence=0.95,
            tags=["family", "homework", "story", "encouragement"],
        )
    )
    store.add_memory(
        MemoryRecord(
            user_id="family001",
            memory_type="feedback",
            content="任务目标：家庭数学作业；执行反馈：分数加减法容易出错，需要继续练习",
            source="seed",
            importance=0.8,
            confidence=0.9,
            tags=["math", "fraction", "feedback"],
        )
    )


def seed_home_service_memories(store: AgentOSStore) -> None:
    records = [
        MemoryRecord(
            user_id="home001",
            memory_type="preference",
            content="用户晚上不喝咖啡，偏好睡前喝温水",
            source="seed",
            importance=0.9,
            confidence=0.95,
            tags=["home", "warm_water", "bedtime"],
        ),
        MemoryRecord(
            user_id="home001",
            memory_type="scene",
            content="常用水杯放在书桌右侧",
            source="seed",
            importance=0.75,
            confidence=0.9,
            tags=["home", "cup", "desk"],
        ),
        MemoryRecord(
            user_id="home001",
            memory_type="feedback",
            content="任务目标：睡前生活辅助；执行反馈：上次提醒太晚，用户希望提前提醒并准备温水",
            source="seed",
            importance=0.8,
            confidence=0.9,
            tags=["home", "warm_water", "feedback"],
        ),
    ]
    for record in records:
        store.add_memory(record)


def evaluate_experiment_suite(store: AgentOSStore | None = None):
    _ = store
    return run_deterministic_suite(load_benchmark())


def evaluate_planners(store: AgentOSStore | None = None) -> dict[str, EvaluationResult]:
    suite = evaluate_experiment_suite(store)
    planner_names = ["no_memory", "preference_only", "feedback_only", "layered_memory"]
    results = {}
    for planner_name in planner_names:
        summaries = {
            summary.metric: summary
            for summary in suite.statistics
            if summary.planner_name == planner_name
            and summary.condition == "clean"
            and summary.scenario_group == "all"
        }
        results[planner_name] = EvaluationResult(
            planner_name=planner_name,
            scenario_count=summaries["preference_match_rate"].sample_count,
            memory_hit_rate=summaries["memory_retrieval_recall"].value,
            preference_match_rate=summaries["preference_match_rate"].value,
            executable_plan_rate=summaries["executable_plan_rate"].value,
            task_completion_rate=summaries["task_coverage_rate"].value,
        )
    return results


def export_evaluation_artifacts(
    store: AgentOSStore | None,
    output_dir: Path,
    stem: str = "agentos-evaluation",
):
    _ = store
    return export_benchmark_artifacts(load_benchmark(), output_dir, stem)
