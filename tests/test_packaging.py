import hashlib
import zipfile
from pathlib import Path

from aimo_interp_infra.packaging import build_submission


def test_build_submission_is_byte_reproducible(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "solution.py").write_text(
        "def are_robust(model_id, problems):\n"
        "    return [False for _ in problems]\n",
        encoding="utf-8",
    )
    (source / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    hash1 = build_submission(source, first, small=False)
    hash2 = build_submission(source, second, small=False)
    assert first.read_bytes() == second.read_bytes()
    assert hash1 == hash2 == hashlib.sha256(first.read_bytes()).hexdigest()


def test_small_track_adds_root_marker_without_mutating_source(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "solution.py").write_text(
        "def are_robust(model_id, problems):\n"
        "    return [True for _ in problems]\n",
        encoding="utf-8",
    )
    destination = tmp_path / "small.zip"
    build_submission(source, destination, small=True)
    assert not (source / "small.txt").exists()
    with zipfile.ZipFile(destination) as archive:
        assert "solution.py" in archive.namelist()
        assert "small.txt" in archive.namelist()
        assert archive.read("small.txt") == b""


def test_build_rejects_missing_solution(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    try:
        build_submission(source, tmp_path / "bad.zip", small=False)
    except ValueError as exc:
        assert "solution.py" in str(exc)
    else:
        raise AssertionError("missing solution.py must be rejected")

