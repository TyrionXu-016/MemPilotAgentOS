from typer.testing import CliRunner

from agentos.cli import app


def test_export_results_accepts_scaled_benchmark_options(tmp_path):
    output_dir = tmp_path / "results"
    database = tmp_path / "unused.sqlite"

    result = CliRunner().invoke(
        app,
        [
            "export-results",
            "--db",
            str(database),
            "--dataset",
            "experiments/data/agentos-benchmark-v1.json",
            "--output-dir",
            str(output_dir),
            "--stem",
            "agentos-evaluation",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "agentos-evaluation-cases.csv").exists()
    assert not database.exists()
