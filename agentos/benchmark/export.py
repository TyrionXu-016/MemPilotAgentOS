from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from agentos.benchmark.models import (
    BenchmarkExperimentResult,
    BenchmarkSuite,
    CaseRunResult,
    ExperimentManifest,
    MetricSummary,
)
from agentos.benchmark.runner import run_benchmark


def benchmark_sha256(benchmark: BenchmarkSuite) -> str:
    canonical = json.dumps(benchmark.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def export_benchmark_artifacts(
    benchmark: BenchmarkSuite,
    output_dir: Path | str,
    stem: str = "agentos-evaluation",
    include_deepseek: bool = False,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_repeats: int = 5,
    concurrency: int = 5,
    cache_path: Path | str | None = None,
) -> dict[str, Path]:
    result = run_benchmark(
        benchmark,
        include_deepseek=include_deepseek,
        deepseek_model=deepseek_model,
        deepseek_repeats=deepseek_repeats,
        concurrency=concurrency,
        cache_path=cache_path,
    )
    manifest = _build_manifest(benchmark, result, deepseek_model, deepseek_repeats)
    result = result.model_copy(update={"manifest": manifest})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": output / f"{stem}.md",
        "csv": output / f"{stem}.csv",
        "scenario_csv": output / f"{stem}-scenarios.csv",
        "robustness_csv": output / f"{stem}-robustness.csv",
        "feedback_loop_csv": output / f"{stem}-feedback-loop.csv",
        "cases_csv": output / f"{stem}-cases.csv",
        "statistics_csv": output / f"{stem}-statistics.csv",
        "significance_csv": output / f"{stem}-significance.csv",
        "deepseek_csv": output / f"{stem}-deepseek.csv",
        "manifest": output / f"{stem}-manifest.json",
        "svg": output / f"{stem}.svg",
        "overall_svg": output / "agentos-overall.svg",
        "ablation_svg": output / "agentos-ablation.svg",
        "robustness_svg": output / "agentos-robustness.svg",
        "confidence_svg": output / "agentos-confidence-intervals.svg",
        "deepseek_svg": output / "agentos-deepseek-baseline.svg",
    }
    clean_summaries = _select_summaries(result, "clean", "all")
    scenario_summaries = [
        summary
        for summary in result.statistics
        if summary.condition == "clean" and summary.scenario_group != "all"
    ]
    robustness_summaries = [
        summary
        for summary in result.statistics
        if summary.planner_name == "layered_memory"
        and summary.condition in {
            "clean",
            "irrelevant_noise",
            "conflict_low_confidence",
            "stale_low_importance",
        }
        and summary.scenario_group == "all"
    ]
    feedback_summaries = [
        summary
        for summary in result.statistics
        if summary.condition in {"before_feedback_writeback", "after_feedback_writeback"}
        and summary.scenario_group == "all"
    ]
    all_runs = result.clean_runs + result.robustness_runs + result.feedback_runs + result.deepseek_runs
    _write(paths["markdown"], _render_markdown(result))
    _write(paths["csv"], _wide_summary_csv(clean_summaries))
    _write(paths["scenario_csv"], _long_summary_csv(scenario_summaries))
    _write(paths["robustness_csv"], _long_summary_csv(robustness_summaries))
    _write(paths["feedback_loop_csv"], _long_summary_csv(feedback_summaries))
    _write(paths["cases_csv"], _model_csv(all_runs))
    _write(paths["statistics_csv"], _model_csv(result.statistics))
    _write(paths["significance_csv"], _model_csv(result.significance))
    _write(paths["deepseek_csv"], _model_csv(result.deepseek_runs, model=CaseRunResult))
    _write(paths["manifest"], manifest.model_dump_json(indent=2) + "\n")
    overall_svg = _render_bar_chart(clean_summaries, "AgentOS 总体规划指标")
    _write(paths["svg"], overall_svg)
    _write(paths["overall_svg"], overall_svg)
    _write(paths["ablation_svg"], _render_bar_chart(clean_summaries, "AgentOS 消融实验指标"))
    _write(paths["robustness_svg"], _render_bar_chart(robustness_summaries, "AgentOS 鲁棒性实验指标"))
    _write(paths["confidence_svg"], _render_confidence_chart(clean_summaries))
    deepseek_summaries = [
        summary for summary in clean_summaries if summary.planner_name.startswith("deepseek_")
    ]
    _write(paths["deepseek_svg"], _render_bar_chart(deepseek_summaries, "AgentOS DeepSeek 对照"))
    return paths


def _build_manifest(
    benchmark: BenchmarkSuite,
    result: BenchmarkExperimentResult,
    model: str,
    repeats: int,
) -> ExperimentManifest:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    fingerprints = sorted(
        {run.system_fingerprint for run in result.deepseek_runs if run.system_fingerprint}
    )
    prompt_hashes = sorted({run.prompt_hash for run in result.deepseek_runs if run.prompt_hash})
    return ExperimentManifest(
        dataset_version=benchmark.version,
        dataset_sha256=benchmark_sha256(benchmark),
        git_commit=commit,
        python_version=platform.python_version(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        deepseek_model=model if result.deepseek_runs else None,
        deepseek_repeats=repeats if result.deepseek_runs else 0,
        system_fingerprints=fingerprints,
        prompt_hashes=prompt_hashes,
    )


def _render_markdown(result: BenchmarkExperimentResult) -> str:
    clean = _select_summaries(result, "clean", "all")
    rows = _wide_summary_rows(clean)
    lines = [
        "# AgentOS 规模化记忆增强规划实验结果",
        "",
        "本报告由 72 条静态基准数据自动生成，覆盖学习陪伴、家庭教育和家庭服务。",
        "任务覆盖率是模拟环境中的规划覆盖代理指标，不代表真实机器人硬件完成率。",
        "",
        "## 规模化场景泛化实验",
        "",
        "| Planner | 样例数 | 记忆召回率 | 偏好匹配率 | 任务覆盖率 | 可执行率 | 技能链正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {planner_name} | {sample_count} | {memory_retrieval_recall:.2f} | "
            "{preference_match_rate:.2f} | {task_coverage_rate:.2f} | "
            "{executable_plan_rate:.2f} | {skill_chain_accuracy:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 消融与鲁棒性实验",
            "",
            "完整分层记忆与无记忆、仅偏好记忆、仅反馈记忆进行成对比较；鲁棒性条件覆盖无关噪声、低置信冲突和低重要度历史记录。",
            "",
            "## 反馈闭环实验",
            "",
            "18 组隐式任务分别比较反馈写回前后规划，验证执行反馈是否改变下一轮任务覆盖。",
            "",
            "## 统计显著性",
            "",
            "比例指标报告 Wilson 95% 置信区间，成对结果使用精确 McNemar 检验并进行 Holm 校正。",
            "",
            "## DeepSeek 对照实验",
            "",
        ]
    )
    if result.deepseek_runs:
        lines.append(
            f"DeepSeek clean 基线已完成 {len(result.deepseek_runs)} 次请求，并按每个样例多数表决进行显著性比较。"
        )
    else:
        lines.append("本次离线导出未启用 DeepSeek；提供 API Key 后可生成 720 次在线对照记录。")
    return "\n".join(lines) + "\n"


def _select_summaries(
    result: BenchmarkExperimentResult,
    condition: str,
    scenario_group: str,
) -> list[MetricSummary]:
    return [
        summary
        for summary in result.statistics
        if summary.condition == condition and summary.scenario_group == scenario_group
    ]


def _wide_summary_rows(summaries: list[MetricSummary]) -> list[dict]:
    rows: dict[str, dict] = {}
    for summary in summaries:
        row = rows.setdefault(
            summary.planner_name,
            {"planner_name": summary.planner_name, "sample_count": summary.sample_count},
        )
        row[summary.metric] = summary.value
    return [rows[name] for name in sorted(rows)]


def _wide_summary_csv(summaries: list[MetricSummary]) -> str:
    fields = [
        "planner_name",
        "sample_count",
        "memory_retrieval_precision",
        "memory_retrieval_recall",
        "preference_match_rate",
        "task_coverage_rate",
        "executable_plan_rate",
        "skill_chain_accuracy",
        "conflict_selection_accuracy",
        "api_success_rate",
    ]
    return _dict_csv(_wide_summary_rows(summaries), fields)


def _long_summary_csv(summaries: list[MetricSummary]) -> str:
    return _model_csv(summaries, model=MetricSummary)


def _model_csv(items, model=None) -> str:
    if items:
        rows = [item.model_dump(mode="json") for item in items]
        fields = list(rows[0])
    elif model is not None:
        rows = []
        fields = list(model.model_fields)
    else:
        return ""
    if "ci_low" in fields and "ci_high" in fields:
        fields.remove("ci_high")
        fields.insert(fields.index("ci_low") + 1, "ci_high")
    return _dict_csv(rows, fields)


def _dict_csv(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _render_bar_chart(summaries: list[MetricSummary], title: str) -> str:
    rows = _wide_summary_rows(summaries)
    metrics = ["preference_match_rate", "task_coverage_rate", "executable_plan_rate"]
    width = max(760, 180 + len(rows) * 110)
    height = 360
    parts = _svg_header(width, height, title)
    for metric_index, metric in enumerate(metrics):
        base_x = 110 + metric_index * 210
        parts.append(f'<text x="{base_x}" y="300" font-size="12">{escape(metric)}</text>')
        for planner_index, row in enumerate(rows):
            value = float(row.get(metric, 0.0))
            bar_height = 200 * value
            x = base_x + planner_index * 30
            y = 260 - bar_height
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="22" height="{bar_height:.1f}" fill="{_color(planner_index)}"/>')
    for index, row in enumerate(rows):
        parts.append(f'<text x="{70 + index * 150}" y="340" font-size="11">{escape(row["planner_name"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _render_confidence_chart(summaries: list[MetricSummary]) -> str:
    selected = [summary for summary in summaries if summary.metric == "task_coverage_rate"]
    parts = _svg_header(820, 360, "AgentOS 95% 置信区间")
    for index, summary in enumerate(selected):
        y = 80 + index * 55
        x1 = 180 + summary.ci_low * 540
        x2 = 180 + summary.ci_high * 540
        x = 180 + summary.value * 540
        parts.append(f'<text x="20" y="{y + 4}" font-size="12">{escape(summary.planner_name)}</text>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#374151" stroke-width="3"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="5" fill="#2563eb"/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">{escape(title)}</text>',
    ]


def _color(index: int) -> str:
    return ["#6b7280", "#0891b2", "#f59e0b", "#2563eb", "#16a34a", "#dc2626"][index % 6]


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
