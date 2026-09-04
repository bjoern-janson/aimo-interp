import json
import subprocess
from pathlib import Path

import pytest

from aimo_interp_infra.upstream import (
    UpstreamLock,
    UpstreamMismatch,
    verify_checkout,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_lock_parses_exact_commit(tmp_path: Path):
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "repository": "https://example.test/repo.git",
                "github_repository": "owner/repo",
                "default_branch": "main",
                "commit": "a" * 40,
                "authority": "PINNED_EXECUTION_CONTRACT",
                "acquired_at_utc": "2026-09-04T12:00:00Z",
                "references": ["https://example.test/repo"],
                "gate_artifact_registry": "RELEASE_REGISTRY.json",
            }
        ),
        encoding="utf-8",
    )
    lock = UpstreamLock.from_json(path)
    assert lock.commit == "a" * 40
    assert lock.github_repository == "owner/repo"
    assert lock.gate_artifact_registry == "RELEASE_REGISTRY.json"


def test_verify_checkout_accepts_matching_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    head = _git(repo, "rev-parse", "HEAD")

    lock = UpstreamLock(
        repository="https://example.test/repo.git",
        github_repository="owner/repo",
        default_branch="main",
        commit=head,
        authority="PINNED_EXECUTION_CONTRACT",
    )
    verify_checkout(lock, repo)


def test_verify_checkout_rejects_wrong_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")

    lock = UpstreamLock(
        repository="https://example.test/repo.git",
        github_repository="owner/repo",
        default_branch="main",
        commit="b" * 40,
        authority="PINNED_EXECUTION_CONTRACT",
    )
    with pytest.raises(UpstreamMismatch):
        verify_checkout(lock, repo)

