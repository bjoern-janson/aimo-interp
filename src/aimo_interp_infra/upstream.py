from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class UpstreamMismatch(RuntimeError):
    """Raised when a local upstream checkout does not match the frozen lock."""


@dataclass(frozen=True)
class UpstreamLock:
    repository: str
    github_repository: str
    default_branch: str
    commit: str
    authority: str
    acquired_at_utc: str | None = None
    references: tuple[str, ...] = ()
    gate_artifact_registry: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "UpstreamLock":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            repository=str(payload["repository"]),
            github_repository=str(payload["github_repository"]),
            default_branch=str(payload["default_branch"]),
            commit=str(payload["commit"]),
            authority=str(payload["authority"]),
            acquired_at_utc=str(payload["acquired_at_utc"]),
            references=tuple(str(item) for item in payload["references"]),
            gate_artifact_registry=(
                str(payload["gate_artifact_registry"])
                if payload.get("gate_artifact_registry") is not None
                else None
            ),
        )


def _run_git(*args: str, cwd: Path | None = None) -> str:
    command = ["git"]
    if cwd is not None:
        command += ["-C", str(cwd)]
    command += list(args)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def verify_checkout(lock: UpstreamLock, checkout: Path) -> None:
    head = _run_git("rev-parse", "HEAD", cwd=checkout)
    if head != lock.commit:
        raise UpstreamMismatch(
            f"upstream HEAD {head} does not match frozen commit {lock.commit}"
        )
    dirty = _run_git("status", "--porcelain", cwd=checkout)
    if dirty:
        raise UpstreamMismatch("upstream checkout contains uncommitted changes")


def materialize_checkout(lock: UpstreamLock, destination: Path) -> None:
    if destination.exists():
        verify_checkout(lock, destination)
        return
    if shutil.which("git") is None:
        raise RuntimeError("git executable is required to materialize upstream")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_git("clone", "--filter=blob:none", lock.repository, str(destination))
    _run_git("checkout", "--detach", lock.commit, cwd=destination)
    verify_checkout(lock, destination)

