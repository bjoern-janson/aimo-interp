from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialRunResult:
    accuracy: float
    coverage: float
    invalid_predictions: int
    stdout: str
    stderr: str


def _extract_metrics(stdout: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for start, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if {"accuracy", "coverage", "invalid_predictions"} <= payload.keys():
            return payload
    raise RuntimeError("official runner output contained no score object")


def run_official(
    upstream_checkout: Path,
    solution: Path,
    input_dir: Path,
    reference_dir: Path,
    small: bool,
    uv_executable: str = "uv",
) -> OfficialRunResult:
    command = [
        uv_executable,
        "run",
        str(upstream_checkout / "scripts" / "run_local.py"),
        str(solution),
        "--input-dir",
        str(input_dir),
        "--reference-dir",
        str(reference_dir),
    ]
    if small:
        command.append("--small")

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        cwd=upstream_checkout,
    )
    metrics = _extract_metrics(completed.stdout)
    return OfficialRunResult(
        accuracy=float(metrics["accuracy"]),
        coverage=float(metrics["coverage"]),
        invalid_predictions=int(metrics["invalid_predictions"]),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

