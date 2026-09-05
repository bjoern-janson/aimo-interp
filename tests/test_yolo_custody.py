import hashlib
from pathlib import Path

import pytest

from aimo_interp_yolo.custody import ensure_no_results, verify_artifact_files, verify_custody_shape

EXPECTED = ["Y001-A", "Y001-B", "Y001-C", "Y001-D", "Y001-E", "Y001-F", "CTRL-T", "CTRL-F"]


def base_custody() -> dict[str, object]:
    return {
        "schema": "aimo-interp-yolo-battery-custody/v0.1",
        "battery_id": "YOLO001-B1",
        "protocol_sha256": "a" * 64,
        "protocol_git_blob_sha": "b" * 40,
        "implementation_commit": "c" * 40,
        "upstream_contract_commit": "e98c489a98acb6c833588dca74228bee9782d5dd",
        "build_source_commit": "c" * 40,
        "track": "small",
        "closed_at_utc": "2026-09-04T18:00:00Z",
        "protocol_state": "CLOSED",
        "artifacts": [],
    }


def test_custody_forbids_self_reference_and_mutable_evaluation_state():
    payload = base_custody()
    verify_custody_shape(payload, require_artifacts=False)
    payload["closure_commit"] = "d" * 40
    with pytest.raises(ValueError, match="closure_commit"):
        verify_custody_shape(payload, require_artifacts=False)
    payload = base_custody()
    payload["evaluation_state"] = "UNSCORED"
    with pytest.raises(ValueError, match="evaluation_state"):
        verify_custody_shape(payload, require_artifacts=False)


def test_existing_result_row_blocks_closure(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    ensure_no_results(path)
    path.write_text('{"observed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="results already exist"):
        ensure_no_results(path)


def test_artifact_byte_change_is_detected(tmp_path: Path):
    output = tmp_path / "dist"
    output.mkdir()
    artifacts = []
    for protocol_id in EXPECTED:
        path = output / f"{protocol_id}.zip"
        path.write_bytes(protocol_id.encode("utf-8"))
        artifacts.append({
            "protocol_id": protocol_id,
            "zip_filename": path.name,
            "zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "zip_size_bytes": path.stat().st_size,
            "source_identity": f"yolo/batteries/{protocol_id}",
            "track": "small",
            "small_marker": True,
        })
    custody = base_custody()
    custody["artifacts"] = artifacts
    verify_artifact_files(custody, output, inspect_zip_structure=False)
    (output / "Y001-A.zip").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="ZIP digest mismatch"):
        verify_artifact_files(custody, output, inspect_zip_structure=False)
