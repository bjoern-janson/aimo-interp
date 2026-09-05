from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

from .build import build_battery
from .jsonutil import canonical_json_bytes
from .protocol import EXPECTED_IDS, load_protocol

UPSTREAM_COMMIT = "e98c489a98acb6c833588dca74228bee9782d5dd"
FORBIDDEN_CUSTODY_KEYS = {"closure_commit", "evaluation_state", "accuracy", "rank", "submission_id"}
_CUSTODY_KEYS = {
    "schema", "battery_id", "protocol_sha256", "protocol_git_blob_sha", "implementation_commit",
    "upstream_contract_commit", "build_source_commit", "track", "closed_at_utc", "protocol_state", "artifacts",
}
_ARTIFACT_KEYS = {"protocol_id", "zip_filename", "zip_sha256", "zip_size_bytes", "source_identity", "track", "small_marker"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(path: Path, repo_root: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()


def git_head(repo_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()


def ensure_no_results(path: Path) -> None:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        raise RuntimeError("B1 results already exist")


def verify_custody_shape(payload: dict[str, object], *, require_artifacts: bool) -> None:
    if not isinstance(payload, dict):
        raise ValueError("custody must be an object")
    for key in FORBIDDEN_CUSTODY_KEYS:
        if key in payload:
            raise ValueError(f"forbidden custody key: {key}")
    missing = _CUSTODY_KEYS - payload.keys()
    extra = payload.keys() - _CUSTODY_KEYS
    if missing:
        raise ValueError(f"custody missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"custody unexpected keys: {sorted(extra)}")
    if payload["schema"] != "aimo-interp-yolo-battery-custody/v0.1":
        raise ValueError("invalid custody schema")
    if payload["battery_id"] != "YOLO001-B1" or payload["track"] != "small" or payload["protocol_state"] != "CLOSED":
        raise ValueError("invalid custody identity/state")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be a list")
    if require_artifacts and [a.get("protocol_id") for a in artifacts if isinstance(a, dict)] != list(EXPECTED_IDS):
        raise ValueError("custody artifacts do not match expected ordered IDs")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("artifact must be an object")
        if artifact.keys() != _ARTIFACT_KEYS:
            raise ValueError("artifact keys do not match custody schema")
        if artifact["track"] != "small" or artifact["small_marker"] is not True:
            raise ValueError("invalid artifact track/marker")
        if artifact["source_identity"] != f"yolo/batteries/{artifact['protocol_id']}":
            raise ValueError("invalid artifact source identity")


def verify_artifact_files(custody: dict[str, object], output_dir: Path, *, inspect_zip_structure: bool = True) -> None:
    artifacts = custody["artifacts"]
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        path = output_dir / str(artifact["zip_filename"])
        if not path.is_file():
            raise RuntimeError(f"missing ZIP: {path.name}")
        if sha256_file(path) != artifact["zip_sha256"]:
            raise RuntimeError(f"ZIP digest mismatch: {path.name}")
        if path.stat().st_size != artifact["zip_size_bytes"]:
            raise RuntimeError(f"ZIP size mismatch: {path.name}")
        if inspect_zip_structure:
            with zipfile.ZipFile(path) as archive:
                for name in ("solution.py", "member.json", "small.txt"):
                    if name not in archive.namelist():
                        raise RuntimeError(f"ZIP missing required member: {name}")
                if archive.read("small.txt") != b"":
                    raise RuntimeError("small.txt marker must be empty")


def _read_upstream_commit(lock_path: Path) -> str:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("UPSTREAM_LOCK commit mismatch")
    return str(payload["commit"])


def prepare_closure(
    protocol_path: Path, custody_path: Path, results_path: Path, output_dir: Path, staging_root: Path,
    package_dir: Path, repo_root: Path, implementation_commit: str, closed_at_utc: str,
) -> dict[str, object]:
    protocol = load_protocol(protocol_path)
    if protocol.protocol_state != "OPEN":
        raise RuntimeError("B1 closure requires OPEN protocol")
    if implementation_commit != git_head(repo_root):
        raise RuntimeError("implementation commit does not match HEAD")
    tracked_dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()
    if tracked_dirty:
        raise RuntimeError("tracked worktree must be clean before closure preparation")
    ensure_no_results(results_path)
    if custody_path.exists():
        raise RuntimeError("custody already exists")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", closed_at_utc):
        raise ValueError("closed_at_utc must be second-resolution UTC Z time")
    upstream_commit = _read_upstream_commit(repo_root / "UPSTREAM_LOCK.json")
    closed_protocol = replace(protocol, protocol_state="CLOSED")
    protocol_path.write_bytes(canonical_json_bytes(asdict(closed_protocol)))
    built = build_battery(protocol_path, output_dir, staging_root, package_dir)
    custody = {
        "schema": "aimo-interp-yolo-battery-custody/v0.1", "battery_id": "YOLO001-B1",
        "protocol_sha256": sha256_file(protocol_path), "protocol_git_blob_sha": git_blob_sha(protocol_path, repo_root),
        "implementation_commit": implementation_commit, "upstream_contract_commit": upstream_commit,
        "build_source_commit": implementation_commit, "track": "small", "closed_at_utc": closed_at_utc,
        "protocol_state": "CLOSED",
        "artifacts": [{"protocol_id": a.protocol_id, "zip_filename": a.zip_path.name, "zip_sha256": a.zip_sha256,
                        "zip_size_bytes": a.zip_size_bytes, "source_identity": a.source_identity, "track": "small", "small_marker": True}
                       for a in built.values()],
    }
    verify_custody_shape(custody, require_artifacts=True)
    custody_path.write_bytes(canonical_json_bytes(custody))
    return custody


def _load_custody(custody_path: Path) -> dict[str, object]:
    payload = json.loads(custody_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("custody must be an object")
    return payload


def verify_precommit_closure(protocol_path: Path, custody_path: Path, results_path: Path, output_dir: Path, repo_root: Path) -> None:
    custody = _load_custody(custody_path)
    verify_custody_shape(custody, require_artifacts=True)
    ensure_no_results(results_path)
    if git_head(repo_root) != custody["implementation_commit"]:
        raise RuntimeError("current HEAD does not match custody implementation commit")
    protocol = load_protocol(protocol_path)
    if protocol.protocol_state != "CLOSED" or sha256_file(protocol_path) != custody["protocol_sha256"] or git_blob_sha(protocol_path, repo_root) != custody["protocol_git_blob_sha"]:
        raise RuntimeError("CLOSED protocol does not match custody")
    verify_artifact_files(custody, output_dir)


def verify_committed_closure(protocol_path: Path, custody_path: Path, results_path: Path, repo_root: Path) -> None:
    custody = _load_custody(custody_path)
    verify_custody_shape(custody, require_artifacts=True)
    ensure_no_results(results_path)
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()
    if parent != custody["implementation_commit"]:
        raise RuntimeError("closure commit parent is not implementation_commit")
    if sha256_file(protocol_path) != custody["protocol_sha256"] or git_blob_sha(protocol_path, repo_root) != custody["protocol_git_blob_sha"]:
        raise RuntimeError("committed protocol does not match custody")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        built = build_battery(protocol_path, root / "dist", root / "stage", repo_root / "src" / "aimo_interp_yolo")
        expected = {str(a["protocol_id"]): a for a in custody["artifacts"]}  # type: ignore[union-attr]
        for protocol_id, artifact in built.items():
            record = expected[protocol_id]
            if artifact.zip_sha256 != record["zip_sha256"] or artifact.zip_size_bytes != record["zip_size_bytes"]:
                raise RuntimeError(f"committed rebuild mismatch: {protocol_id}")
