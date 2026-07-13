from __future__ import annotations

import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from agentos.benchmark.models import (
    BenchmarkCase,
    BenchmarkExperimentResult,
    BenchmarkSuite,
    CaseRunResult,
)
from agentos.benchmark.statistics import compare_deepseek, compare_planners, summarize_runs
from agentos.deepseek import DeepSeekConfigurationError, DeepSeekPlanGenerator
from agentos.deepseek_planner import DeepSeekPlanner
from agentos.memory import MemoryManager
from agentos.models import MemoryRecord, Plan
from agentos.ontology import PROFILES
from agentos.planner import LayeredMemoryPlanner, MemoryFilteredPlanner, NoMemoryPlanner
from agentos.retriever import MemoryRetriever
from agentos.skills import build_default_registry
from agentos.storage import AgentOSStore


def run_deterministic_suite(benchmark: BenchmarkSuite) -> BenchmarkExperimentResult:
    clean_runs = _run_clean(benchmark)
    robustness_runs = _run_robustness(benchmark)
    feedback_runs = _run_feedback(benchmark)
    nonduplicate_robustness = [run for run in robustness_runs if run.condition != "clean"]
    all_runs = clean_runs + nonduplicate_robustness + feedback_runs
    return BenchmarkExperimentResult(
        clean_runs=clean_runs,
        robustness_runs=robustness_runs,
        feedback_runs=feedback_runs,
        statistics=summarize_runs(all_runs),
        significance=compare_planners(clean_runs),
    )


class DeepSeekRunCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.runs: dict[str, CaseRunResult] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    run = CaseRunResult.model_validate_json(line)
                    self.runs[run.run_key] = run

    def get(self, run_key: str) -> CaseRunResult | None:
        run = self.runs.get(run_key)
        return run if run and run.success else None

    def append(self, run: CaseRunResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(run.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.runs[run.run_key] = run


def run_deepseek_suite(
    benchmark: BenchmarkSuite,
    model: str = "deepseek-v4-flash",
    repeats: int = 5,
    concurrency: int = 5,
    cache_path: Path | str | None = None,
    generator_factory: Callable[[], DeepSeekPlanGenerator] | None = None,
) -> list[CaseRunResult]:
    if generator_factory is None and not os.environ.get("DEEPSEEK_API_KEY"):
        raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is required for DeepSeek experiments")
    factory = generator_factory or (lambda: DeepSeekPlanGenerator(model=model))
    cache = DeepSeekRunCache(cache_path) if cache_path else None
    runs = []
    jobs = []
    for case in benchmark.cases:
        for planner_name in ("deepseek_no_memory", "deepseek_layered_memory"):
            for repeat_index in range(repeats):
                run_key = f"{benchmark.version}:{case.case_id}:clean:{planner_name}:{repeat_index}:{model}"
                cached = cache.get(run_key) if cache else None
                if cached:
                    runs.append(cached)
                else:
                    jobs.append((case, planner_name, repeat_index, run_key))
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _run_deepseek_job,
                case,
                planner_name,
                repeat_index,
                run_key,
                factory,
            ): run_key
            for case, planner_name, repeat_index, run_key in jobs
        }
        for future in as_completed(futures):
            run = future.result()
            runs.append(run)
            if cache:
                cache.append(run)
    return sorted(runs, key=lambda run: run.run_key)


def run_benchmark(
    benchmark: BenchmarkSuite,
    include_deepseek: bool = False,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_repeats: int = 5,
    concurrency: int = 5,
    cache_path: Path | str | None = None,
) -> BenchmarkExperimentResult:
    result = run_deterministic_suite(benchmark)
    if not include_deepseek:
        return result
    deepseek_runs = run_deepseek_suite(
        benchmark,
        model=deepseek_model,
        repeats=deepseek_repeats,
        concurrency=concurrency,
        cache_path=cache_path,
    )
    nonduplicate_robustness = [
        run for run in result.robustness_runs if run.condition != "clean"
    ]
    all_runs = result.clean_runs + nonduplicate_robustness + result.feedback_runs + deepseek_runs
    return result.model_copy(
        update={
            "deepseek_runs": deepseek_runs,
            "statistics": summarize_runs(all_runs),
            "significance": result.significance + compare_deepseek(deepseek_runs),
        }
    )


def _run_clean(benchmark: BenchmarkSuite) -> list[CaseRunResult]:
    runs = []
    for case in benchmark.cases:
        with TemporaryDirectory() as tmpdir:
            store = _new_store(Path(tmpdir))
            relevant_ids, conflict_ids = _seed_case(store, case, "clean")
            for planner in _build_planners(store):
                runs.append(
                    _run_case(planner, store, case, "clean", relevant_ids, conflict_ids)
                )
    return runs


def _run_robustness(benchmark: BenchmarkSuite) -> list[CaseRunResult]:
    runs = []
    for condition in (
        "clean",
        "irrelevant_noise",
        "conflict_low_confidence",
        "stale_low_importance",
    ):
        for case in benchmark.cases:
            with TemporaryDirectory() as tmpdir:
                store = _new_store(Path(tmpdir))
                relevant_ids, conflict_ids = _seed_case(store, case, condition)
                planner = LayeredMemoryPlanner(build_default_registry(), MemoryRetriever(store))
                runs.append(
                    _run_case(planner, store, case, condition, relevant_ids, conflict_ids)
                )
    return runs


def _run_feedback(benchmark: BenchmarkSuite) -> list[CaseRunResult]:
    implicit_cases = [case for case in benchmark.cases if case.explicitness == "implicit"]
    runs = []
    for case in implicit_cases:
        with TemporaryDirectory() as tmpdir:
            store = _new_store(Path(tmpdir))
            relevant_ids = set()
            for memory in case.memories:
                if memory.memory_type == "preference":
                    added = store.add_memory(memory)
                    if added.id is not None:
                        relevant_ids.add(added.id)
            planner = LayeredMemoryPlanner(build_default_registry(), MemoryRetriever(store))
            runs.append(
                _run_case(
                    planner,
                    store,
                    case,
                    "before_feedback_writeback",
                    relevant_ids,
                    set(),
                    planner_name="before_feedback_writeback",
                )
            )
            feedback = MemoryManager(store).record_feedback(
                case.user_id,
                case.goal,
                f"{case.expected_items[0]} 仍需继续处理",
            )
            if feedback.id is not None:
                relevant_ids.add(feedback.id)
            runs.append(
                _run_case(
                    planner,
                    store,
                    case,
                    "after_feedback_writeback",
                    relevant_ids,
                    set(),
                    planner_name="after_feedback_writeback",
                )
            )
    return runs


def _run_deepseek_job(
    case: BenchmarkCase,
    planner_name: str,
    repeat_index: int,
    run_key: str,
    generator_factory: Callable[[], DeepSeekPlanGenerator],
) -> CaseRunResult:
    with TemporaryDirectory() as tmpdir:
        store = _new_store(Path(tmpdir))
        relevant_ids, conflict_ids = _seed_case(store, case, "clean")
        registry = build_default_registry()
        retriever = MemoryRetriever(store) if planner_name == "deepseek_layered_memory" else None
        planner = DeepSeekPlanner(
            registry,
            generator_factory(),
            planner_name=planner_name,
            retriever=retriever,
        )
        run = _run_case(
            planner,
            store,
            case,
            "clean",
            relevant_ids,
            conflict_ids,
            repeat_index=repeat_index,
        )
        return run.model_copy(update={"run_key": run_key})


def _new_store(directory: Path) -> AgentOSStore:
    store = AgentOSStore(directory / "benchmark.sqlite")
    store.migrate()
    return store


def _build_planners(store: AgentOSStore):
    registry = build_default_registry()
    retriever = MemoryRetriever(store)
    return [
        NoMemoryPlanner(registry),
        MemoryFilteredPlanner(
            registry, retriever, "preference_only", {"profile", "preference"}
        ),
        MemoryFilteredPlanner(
            registry, retriever, "feedback_only", {"task", "feedback"}
        ),
        LayeredMemoryPlanner(registry, retriever),
    ]


def _seed_case(
    store: AgentOSStore,
    case: BenchmarkCase,
    condition: str,
) -> tuple[set[int], set[int]]:
    relevant_ids = set()
    conflict_ids = set()
    for memory in case.memories:
        added = store.add_memory(memory)
        if memory.memory_key in case.relevant_memory_keys and added.id is not None:
            relevant_ids.add(added.id)
    if condition == "irrelevant_noise":
        store.add_memory(
            MemoryRecord(
                user_id="other-user",
                memory_type="preference",
                content="其他用户的高权重无关偏好",
                source="noise",
                importance=1.0,
                confidence=1.0,
                tags=["style:noise"],
            )
        )
        store.add_memory(
            MemoryRecord(
                user_id=case.user_id,
                memory_type="scene",
                content="与当前任务无关的低权重场景",
                source="noise",
                importance=0.1,
                confidence=0.2,
                tags=["noise"],
            )
        )
    elif condition in {"conflict_low_confidence", "stale_low_importance"}:
        alternatives = PROFILES[case.scenario_group]
        style, item = next(
            pair
            for pair in alternatives
            if pair[0] != case.expected_style and pair[1] not in case.expected_items
        )
        importance = 0.2 if condition == "conflict_low_confidence" else 0.05
        confidence = 0.2 if condition == "conflict_low_confidence" else 0.3
        for memory_type, tag, value in (
            ("preference", f"style:{style}", style),
            ("feedback", f"item:{item}", item),
        ):
            added = store.add_memory(
                MemoryRecord(
                    user_id=case.user_id,
                    memory_type=memory_type,
                    content=f"干扰历史记录 {value}",
                    source=condition,
                    importance=importance,
                    confidence=confidence,
                    tags=[f"domain:{case.scenario_group}", tag, condition],
                )
            )
            if added.id is not None:
                conflict_ids.add(added.id)
    return relevant_ids, conflict_ids


def _run_case(
    planner,
    store: AgentOSStore,
    case: BenchmarkCase,
    condition: str,
    relevant_ids: set[int],
    conflict_ids: set[int],
    planner_name: str | None = None,
    repeat_index: int = 0,
) -> CaseRunResult:
    started = time.perf_counter()
    plan = None
    error = None
    executable = False
    try:
        plan = planner.plan(case.user_id, case.goal)
        build_default_registry().validate_plan(plan)
        executable = True
    except Exception as exc:
        error = str(exc)
    latency_ms = (time.perf_counter() - started) * 1000
    return _result_from_plan(
        plan,
        case,
        condition,
        planner_name or planner.planner_name,
        relevant_ids,
        conflict_ids,
        executable,
        error,
        latency_ms,
        repeat_index,
    )


def _result_from_plan(
    plan: Plan | None,
    case: BenchmarkCase,
    condition: str,
    planner_name: str,
    relevant_ids: set[int],
    conflict_ids: set[int],
    executable: bool,
    error: str | None,
    latency_ms: float,
    repeat_index: int,
) -> CaseRunResult:
    retrieved_ids = {item.memory_id for item in plan.evidence if item.memory_id is not None} if plan else set()
    relevant_retrieved = retrieved_ids & relevant_ids
    precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0
    used_ids = set(plan.used_memory_ids) if plan else set()
    metadata = plan.generation_metadata if plan else {}
    return CaseRunResult(
        run_key=f"{case.case_id}:{condition}:{planner_name}:{repeat_index}",
        case_id=case.case_id,
        scenario_group=case.scenario_group,
        explicitness=case.explicitness,
        condition=condition,
        planner_name=planner_name,
        repeat_index=repeat_index,
        success=plan is not None and error is None,
        error=error,
        memory_retrieval_precision=precision,
        memory_retrieval_recall=recall,
        preference_match=bool(plan and plan.context.interaction_style == case.expected_style),
        task_coverage=bool(plan and set(plan.context.task_items) & set(case.expected_items)),
        executable_plan=executable,
        skill_chain_correct=bool(plan and [step.skill for step in plan.steps] == case.required_skills),
        conflict_selection_correct=bool(plan and not (used_ids & conflict_ids)),
        latency_ms=latency_ms,
        model=metadata.get("model"),
        system_fingerprint=metadata.get("system_fingerprint"),
        prompt_hash=metadata.get("prompt_hash"),
        response_id=metadata.get("response_id"),
        prompt_tokens=metadata.get("prompt_tokens", 0),
        completion_tokens=metadata.get("completion_tokens", 0),
    )
