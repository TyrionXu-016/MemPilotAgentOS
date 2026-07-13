from __future__ import annotations

import math
import statistics as stdlib_statistics
from collections import defaultdict

from agentos.benchmark.models import CaseRunResult, MetricSummary, SignificanceResult


RATE_METRICS = {
    "memory_retrieval_precision": lambda run: run.memory_retrieval_precision,
    "memory_retrieval_recall": lambda run: run.memory_retrieval_recall,
    "preference_match_rate": lambda run: float(run.preference_match),
    "task_coverage_rate": lambda run: float(run.task_coverage),
    "executable_plan_rate": lambda run: float(run.executable_plan),
    "skill_chain_accuracy": lambda run: float(run.skill_chain_correct),
    "conflict_selection_accuracy": lambda run: float(run.conflict_selection_correct),
    "api_success_rate": lambda run: float(run.success),
}


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = min(b, c)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    total = len(p_values)
    for rank, (original_index, value) in enumerate(indexed):
        corrected = min(1.0, value * (total - rank))
        previous = max(previous, corrected)
        adjusted[original_index] = previous
    return adjusted


def summarize_runs(runs: list[CaseRunResult]) -> list[MetricSummary]:
    groups: dict[tuple[str, str, str], list[CaseRunResult]] = defaultdict(list)
    for run in runs:
        groups[(run.planner_name, run.condition, "all")].append(run)
        groups[(run.planner_name, run.condition, run.scenario_group)].append(run)
    summaries = []
    for (planner, condition, scenario_group), grouped_runs in sorted(groups.items()):
        for metric, accessor in RATE_METRICS.items():
            values = [accessor(run) for run in grouped_runs]
            low, high = wilson_interval(sum(values), len(values))
            summaries.append(
                MetricSummary(
                    planner_name=planner,
                    condition=condition,
                    scenario_group=scenario_group,
                    metric=metric,
                    sample_count=len(values),
                    value=sum(values) / len(values),
                    standard_deviation=stdlib_statistics.pstdev(values) if len(values) > 1 else 0.0,
                    ci_low=low,
                    ci_high=high,
                )
            )
        latencies = [run.latency_ms for run in grouped_runs]
        latency_stddev = stdlib_statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        for metric, value in (
            ("latency_ms_mean", sum(latencies) / len(latencies)),
            ("latency_ms_p50", _percentile(latencies, 0.50)),
            ("latency_ms_p95", _percentile(latencies, 0.95)),
        ):
            summaries.append(
                MetricSummary(
                    planner_name=planner,
                    condition=condition,
                    scenario_group=scenario_group,
                    metric=metric,
                    sample_count=len(latencies),
                    value=value,
                    standard_deviation=latency_stddev,
                    ci_low=0.0,
                    ci_high=0.0,
                )
            )
    return summaries


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def compare_planners(clean_runs: list[CaseRunResult]) -> list[SignificanceResult]:
    by_planner = defaultdict(dict)
    for run in clean_runs:
        by_planner[run.planner_name][run.case_id] = run
    comparisons = ["no_memory", "preference_only", "feedback_only"]
    pending = []
    for baseline in comparisons:
        for metric, attribute in (
            ("preference_match_rate", "preference_match"),
            ("task_coverage_rate", "task_coverage"),
        ):
            baseline_runs = by_planner[baseline]
            layered_runs = by_planner["layered_memory"]
            case_ids = sorted(set(baseline_runs) & set(layered_runs))
            b = sum(
                bool(getattr(baseline_runs[case_id], attribute))
                and not bool(getattr(layered_runs[case_id], attribute))
                for case_id in case_ids
            )
            c = sum(
                not bool(getattr(baseline_runs[case_id], attribute))
                and bool(getattr(layered_runs[case_id], attribute))
                for case_id in case_ids
            )
            pending.append((baseline, metric, len(case_ids), b, c, exact_mcnemar(b, c)))
    adjusted = holm_adjust([item[-1] for item in pending])
    return [
        SignificanceResult(
            baseline_planner=baseline,
            comparison_planner="layered_memory",
            metric=metric,
            sample_count=count,
            baseline_only_successes=b,
            comparison_only_successes=c,
            p_value=p_value,
            adjusted_p_value=adjusted_p,
        )
        for (baseline, metric, count, b, c, p_value), adjusted_p in zip(pending, adjusted)
    ]


def compare_deepseek(runs: list[CaseRunResult]) -> list[SignificanceResult]:
    by_planner = defaultdict(lambda: defaultdict(list))
    for run in runs:
        by_planner[run.planner_name][run.case_id].append(run)
    if not {"deepseek_no_memory", "deepseek_layered_memory"} <= set(by_planner):
        return []
    pending = []
    for metric, attribute in (
        ("preference_match_rate", "preference_match"),
        ("task_coverage_rate", "task_coverage"),
    ):
        baseline_cases = by_planner["deepseek_no_memory"]
        comparison_cases = by_planner["deepseek_layered_memory"]
        case_ids = sorted(set(baseline_cases) & set(comparison_cases))

        def majority(case_runs):
            return sum(bool(getattr(run, attribute)) for run in case_runs) > len(case_runs) / 2

        b = sum(
            majority(baseline_cases[case_id]) and not majority(comparison_cases[case_id])
            for case_id in case_ids
        )
        c = sum(
            not majority(baseline_cases[case_id]) and majority(comparison_cases[case_id])
            for case_id in case_ids
        )
        pending.append((metric, len(case_ids), b, c, exact_mcnemar(b, c)))
    adjusted = holm_adjust([item[-1] for item in pending])
    return [
        SignificanceResult(
            baseline_planner="deepseek_no_memory",
            comparison_planner="deepseek_layered_memory",
            metric=metric,
            sample_count=count,
            baseline_only_successes=b,
            comparison_only_successes=c,
            p_value=p_value,
            adjusted_p_value=adjusted_p,
        )
        for (metric, count, b, c, p_value), adjusted_p in zip(pending, adjusted)
    ]
