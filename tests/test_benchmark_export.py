import json

from agentos.benchmark.dataset import load_benchmark
from agentos.benchmark.export import export_benchmark_artifacts


def test_export_writes_paper_ready_benchmark_artifacts(tmp_path):
    paths = export_benchmark_artifacts(load_benchmark(), tmp_path, "agentos-evaluation")

    expected = {
        "markdown",
        "csv",
        "scenario_csv",
        "robustness_csv",
        "feedback_loop_csv",
        "cases_csv",
        "statistics_csv",
        "significance_csv",
        "deepseek_csv",
        "manifest",
        "svg",
        "overall_svg",
        "ablation_svg",
        "robustness_svg",
        "confidence_svg",
        "deepseek_svg",
    }
    assert expected <= set(paths)
    assert all(paths[name].exists() for name in expected)

    markdown = paths["markdown"].read_text(encoding="utf-8")
    cases = paths["cases_csv"].read_text(encoding="utf-8").splitlines()
    statistics = paths["statistics_csv"].read_text(encoding="utf-8")
    significance = paths["significance_csv"].read_text(encoding="utf-8")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert "规模化场景泛化实验" in markdown
    assert "统计显著性" in markdown
    assert "DeepSeek 对照实验" in markdown
    assert len(cases) == 1 + 288 + 288 + 36
    assert "ci_low,ci_high" in statistics
    assert "adjusted_p_value" in significance
    assert manifest["dataset_version"] == "1.0"
    assert len(manifest["dataset_sha256"]) == 64
    assert "AgentOS 95% 置信区间" in paths["confidence_svg"].read_text(encoding="utf-8")
    assert "AgentOS DeepSeek 对照" in paths["deepseek_svg"].read_text(encoding="utf-8")
