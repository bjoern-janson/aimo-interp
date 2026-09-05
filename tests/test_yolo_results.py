from pathlib import Path
import json

import pytest

from aimo_interp_yolo.results import ResultEvent, append_result, derive_evaluation_state, validate_event

IDS = ["Y001-A", "Y001-B", "Y001-C", "Y001-D", "Y001-E", "Y001-F", "CTRL-T", "CTRL-F"]


def event(protocol_id: str, attempt: int = 1, status: str = "SCORED") -> ResultEvent:
    scored = status == "SCORED"
    return ResultEvent("aimo-interp-yolo-result-event/v0.1", "YOLO001-B1", protocol_id, "a" * 64, "b" * 40,
                       str(900000 + IDS.index(protocol_id)), attempt, "2026-09-04T18:00:00Z", "2026-09-04T18:10:00Z",
                       status, 0.5 if scored else None, 1.0 if scored else None, 0 if scored else None, None, None, "")


def test_evaluation_state_is_derived_from_first_attempts():
    assert derive_evaluation_state(IDS, []) == "UNSCORED"
    assert derive_evaluation_state(IDS, [event("Y001-A")]) == "PARTIAL"
    assert derive_evaluation_state(IDS, [event(i) for i in IDS]) == "COMPLETE"
    assert derive_evaluation_state(IDS, [event(i) for i in IDS] + [event("Y001-A", 2)]) == "COMPLETE"


def test_scored_event_requires_all_metrics():
    bad = event("Y001-A")
    bad = ResultEvent(**{**bad.__dict__, "coverage": None})
    with pytest.raises(ValueError, match="SCORED requires metrics"):
        validate_event(bad)


def test_failure_event_forbids_metrics():
    bad = event("Y001-A", status="INFRASTRUCTURE_FAILED")
    bad = ResultEvent(**{**bad.__dict__, "accuracy": 0.5})
    with pytest.raises(ValueError, match="non-SCORED forbids metrics"):
        validate_event(bad)


def test_append_rejects_duplicate_and_digest_mismatch(tmp_path: Path):
    protocol = json.loads((Path(__file__).parents[1] / "yolo/batteries/YOLO001-B1.protocol.json").read_text())
    protocol["protocol_state"] = "CLOSED"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    zip_path = tmp_path / "Y001-A.zip"
    zip_path.write_bytes(b"a")
    custody = {"artifacts": [{"protocol_id": "Y001-A", "zip_sha256": "a" * 64}]}
    custody_path = tmp_path / "custody.json"
    custody_path.write_text(json.dumps(custody), encoding="utf-8")
    # digest fixture is intentionally tied to event below
    import hashlib
    custody["artifacts"][0]["zip_sha256"] = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    custody_path.write_text(json.dumps(custody), encoding="utf-8")
    results = tmp_path / "results.jsonl"
    e = event("Y001-A")
    e = ResultEvent(**{**e.__dict__, "zip_sha256": custody["artifacts"][0]["zip_sha256"]})
    append_result(protocol_path, custody_path, results, e)
    with pytest.raises(ValueError, match="duplicate result attempt"):
        append_result(protocol_path, custody_path, results, e)
    bad = ResultEvent(**{**e.__dict__, "zip_sha256": "b" * 64})
    with pytest.raises(ValueError, match="ZIP digest mismatch"):
        append_result(protocol_path, custody_path, results, bad)
