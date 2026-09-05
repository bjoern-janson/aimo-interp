import hashlib
import json
import zipfile
from pathlib import Path

from aimo_interp_yolo.build import build_battery

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "yolo" / "batteries" / "YOLO001-B1.protocol.json"
PACKAGE = ROOT / "src" / "aimo_interp_yolo"
EXPECTED = ["Y001-A", "Y001-B", "Y001-C", "Y001-D", "Y001-E", "Y001-F", "CTRL-T", "CTRL-F"]


def test_battery_build_is_byte_reproducible_and_has_logical_source_identity(tmp_path: Path):
    first = build_battery(PROTOCOL, tmp_path / "out1", tmp_path / "stage1", PACKAGE)
    second = build_battery(PROTOCOL, tmp_path / "out2", tmp_path / "stage2", PACKAGE)
    assert list(first) == list(second) == EXPECTED
    for protocol_id in EXPECTED:
        a, b = first[protocol_id], second[protocol_id]
        assert a.zip_path.read_bytes() == b.zip_path.read_bytes()
        assert a.zip_sha256 == hashlib.sha256(a.zip_path.read_bytes()).hexdigest()
        assert a.source_identity == b.source_identity == f"yolo/batteries/{protocol_id}"


def test_every_zip_is_small_track_and_self_contained(tmp_path: Path):
    built = build_battery(PROTOCOL, tmp_path / "out", tmp_path / "stage", PACKAGE)
    required = {
        "solution.py", "member.json", "small.txt",
        "aimo_interp_yolo/__init__.py", "aimo_interp_yolo/jsonutil.py",
        "aimo_interp_yolo/protocol.py", "aimo_interp_yolo/core.py",
        "aimo_interp_yolo/model_runtime.py", "aimo_interp_yolo/submission.py",
    }
    for artifact in built.values():
        with zipfile.ZipFile(artifact.zip_path) as archive:
            assert required <= set(archive.namelist())
            assert archive.read("small.txt") == b""


def test_battery_build_removes_stale_zip_without_removing_unrelated_output_files(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    stale_zip = output_dir / "old-submission.zip"
    stale_zip.write_bytes(b"stale")
    notes = output_dir / "notes.txt"
    notes.write_text("preserve me", encoding="utf-8")

    build_battery(PROTOCOL, output_dir, tmp_path / "stage", PACKAGE)

    assert {path.name for path in output_dir.glob("*.zip")} == {
        "yolo001-b1-y001-a-small.zip",
        "yolo001-b1-y001-b-small.zip",
        "yolo001-b1-y001-c-small.zip",
        "yolo001-b1-y001-d-small.zip",
        "yolo001-b1-y001-e-small.zip",
        "yolo001-b1-y001-f-small.zip",
        "yolo001-b1-ctrl-t-small.zip",
        "yolo001-b1-ctrl-f-small.zip",
    }
    assert not stale_zip.exists()
    assert notes.read_text(encoding="utf-8") == "preserve me"


def test_y001_d_member_payload_is_exact_and_source_has_no_marker(tmp_path: Path):
    built = build_battery(PROTOCOL, tmp_path / "out", tmp_path / "stage", PACKAGE)
    artifact = built["Y001-D"]
    with zipfile.ZipFile(artifact.zip_path) as archive:
        payload = json.loads(archive.read("member.json"))
    assert payload["member"]["temperature_schedule"] == [0.4, 0.4, 0.4]
    assert payload["member"]["seed_indices"] == [1, 2, 3]
    assert not (tmp_path / "stage" / "Y001-D" / "small.txt").exists()
