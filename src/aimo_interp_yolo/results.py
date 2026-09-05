from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .jsonutil import canonical_json_bytes
from .protocol import load_protocol

TERMINAL = {"SCORED", "INFRASTRUCTURE_FAILED", "WITHDRAWN_BEFORE_UPLOAD"}


@dataclass(frozen=True)
class ResultEvent:
    schema: str
    battery_id: str
    protocol_id: str
    zip_sha256: str
    closure_commit_sha: str
    codabench_submission_id: str
    attempt_index: int
    submitted_at_utc: str
    observed_at_utc: str
    execution_status: str
    accuracy: float | None
    coverage: float | None
    invalid_predictions: int | None
    rank_if_observed: int | None
    leaderboard_snapshot_context: str | None
    notes: str


def read_events(path: Path) -> list[ResultEvent]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    return [ResultEvent(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_event(event: ResultEvent) -> None:
    if event.schema != "aimo-interp-yolo-result-event/v0.1" or event.battery_id != "YOLO001-B1":
        raise ValueError("invalid result event identity")
    if event.attempt_index < 1:
        raise ValueError("attempt_index must be >= 1")
    if event.execution_status not in TERMINAL:
        raise ValueError("invalid execution status")
    if event.execution_status == "SCORED":
        if event.accuracy is None or event.coverage is None or event.invalid_predictions is None:
            raise ValueError("SCORED requires metrics")
    elif event.accuracy is not None or event.coverage is not None or event.invalid_predictions is not None:
        raise ValueError("non-SCORED forbids metrics")


def derive_evaluation_state(expected_ids: list[str] | tuple[str, ...], events: list[ResultEvent]) -> str:
    if not events:
        return "UNSCORED"
    first_terminal = {e.protocol_id for e in events if e.attempt_index == 1 and e.execution_status in TERMINAL}
    return "COMPLETE" if first_terminal == set(expected_ids) else "PARTIAL"


def append_result(protocol_path: Path, custody_path: Path, results_path: Path, event: ResultEvent) -> None:
    protocol = load_protocol(protocol_path)
    if protocol.protocol_state != "CLOSED":
        raise RuntimeError("results require CLOSED protocol")
    custody = json.loads(custody_path.read_text(encoding="utf-8"))
    artifact = next((a for a in custody["artifacts"] if a["protocol_id"] == event.protocol_id), None)
    if artifact is None:
        raise ValueError("unknown protocol_id")
    if artifact["zip_sha256"] != event.zip_sha256:
        raise ValueError("ZIP digest mismatch")
    existing = read_events(results_path)
    if any(e.protocol_id == event.protocol_id and e.attempt_index == event.attempt_index for e in existing):
        raise ValueError("duplicate result attempt")
    validate_event(event)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("ab") as stream:
        stream.write(canonical_json_bytes(asdict(event)))
