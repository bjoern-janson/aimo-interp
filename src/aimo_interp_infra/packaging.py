from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _eligible_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source_dir).as_posix())


def _write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def build_submission(source_dir: Path, destination: Path, small: bool) -> str:
    if not (source_dir / "solution.py").is_file():
        raise ValueError("submission source must contain root solution.py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for path in _eligible_files(source_dir):
            relative = path.relative_to(source_dir).as_posix()
            if relative.casefold() == "small.txt":
                continue
            _write_bytes(archive, relative, path.read_bytes())
        if small:
            _write_bytes(archive, "small.txt", b"")
    return hashlib.sha256(destination.read_bytes()).hexdigest()

