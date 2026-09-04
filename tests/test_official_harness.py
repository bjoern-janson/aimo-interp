from pathlib import Path

from aimo_interp_infra.official_harness import run_official


def test_run_official_parses_metrics_from_stdout(tmp_path: Path):
    upstream = tmp_path / "upstream"
    scripts = upstream / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_local.py").write_text(
        "import json\n"
        "print(json.dumps({'accuracy': 0.5, 'coverage': 1.0, "
        "'invalid_predictions': 0}))\n",
        encoding="utf-8",
    )
    solution = tmp_path / "solution"
    solution.mkdir()
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    input_dir.mkdir()
    reference_dir.mkdir()

    result = run_official(
        upstream_checkout=upstream,
        solution=solution,
        input_dir=input_dir,
        reference_dir=reference_dir,
        small=False,
        uv_executable="uv",
    )
    assert result.accuracy == 0.5
    assert result.coverage == 1.0
    assert result.invalid_predictions == 0


def test_run_official_propagates_runner_failure(tmp_path: Path):
    upstream = tmp_path / "upstream"
    scripts = upstream / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_local.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    solution = tmp_path / "solution"
    solution.mkdir()
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    input_dir.mkdir()
    reference_dir.mkdir()

    import subprocess

    try:
        run_official(
            upstream_checkout=upstream,
            solution=solution,
            input_dir=input_dir,
            reference_dir=reference_dir,
            small=False,
            uv_executable="uv",
        )
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 7
    else:
        raise AssertionError("official runner failure must propagate")


def test_run_official_forwards_small_flag(tmp_path: Path):
    upstream = tmp_path / "upstream"
    scripts = upstream / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_local.py").write_text(
        "import json, sys\n"
        "assert '--small' in sys.argv\n"
        "print(json.dumps({'accuracy': 1.0, 'coverage': 1.0, "
        "'invalid_predictions': 0}))\n",
        encoding="utf-8",
    )
    solution = tmp_path / "solution"
    solution.mkdir()
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    input_dir.mkdir()
    reference_dir.mkdir()

    result = run_official(
        upstream_checkout=upstream,
        solution=solution,
        input_dir=input_dir,
        reference_dir=reference_dir,
        small=True,
        uv_executable="uv",
    )
    assert result.accuracy == 1.0

