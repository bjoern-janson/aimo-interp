import hashlib
import json
from pathlib import Path

import pytest

from aimo_interp_infra.release_gate import (
    ArtifactAlreadyRegistered,
    canonical_directory_manifest,
    record_artifact,
    register,
)


def make_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "aimo-interp-release-registry/v0.2",
                "training_data": None,
                "cot_activation_interface": None,
                "gate_open": False,
            }
        ),
        encoding="utf-8",
    )


def test_directory_identity_is_exact_canonical_json_manifest(tmp_path: Path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "z.txt").write_bytes(b"z")
    (root / "docs").mkdir()
    (root / "docs" / "a.txt").write_bytes(b"abc")

    expected = [
        {
            "path": "docs/a.txt",
            "size_bytes": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        },
        {
            "path": "z.txt",
            "size_bytes": 1,
            "sha256": hashlib.sha256(b"z").hexdigest(),
        },
    ]
    assert canonical_directory_manifest(root) == expected
    record = record_artifact(
        root, "https://example.test/a", "r1", "2026-09-04T12:00:00Z"
    )
    encoded = json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert record.sha256 == hashlib.sha256(encoded).hexdigest()
    assert record.identity_algorithm == "sha256-canonical-json-manifest/v1"


def test_file_identity_is_exact_file_bytes(tmp_path: Path):
    path = tmp_path / "train.json"
    path.write_bytes(b"{}")
    record = record_artifact(
        path, "https://example.test/t", "r1", "2026-09-04T12:00:00Z"
    )
    assert record.sha256 == hashlib.sha256(b"{}").hexdigest()
    assert record.path_kind == "file"


def test_gate_opens_after_both_records_and_never_replaces_one(tmp_path: Path):
    registry = tmp_path / "registry.json"
    make_registry(registry)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    first_state = register(
        registry,
        "training_data",
        record_artifact(first, "https://example.test/t", "t", "2026-09-04T12:00:00Z"),
    )
    assert first_state["gate_open"] is False
    second_state = register(
        registry,
        "cot_activation_interface",
        record_artifact(second, "https://example.test/z", "z", "2026-09-04T12:01:00Z"),
    )
    assert second_state["gate_open"] is True
    with pytest.raises(ArtifactAlreadyRegistered):
        register(
            registry,
            "training_data",
            record_artifact(first, "https://example.test/t", "t", "2026-09-04T12:00:00Z"),
        )

