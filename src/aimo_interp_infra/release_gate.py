from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


class ArtifactAlreadyRegistered(RuntimeError):
    """The registry API never replaces a filled artifact slot."""


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    revision: str
    acquired_at_utc: str
    sha256: str
    size_bytes: int
    path_name: str
    path_kind: str
    identity_algorithm: str


def canonical_directory_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_dir():
        raise NotADirectoryError(path)
    result: list[dict[str, object]] = []
    for file_path in sorted(
        (p for p in path.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(path).as_posix(),
    ):
        payload = file_path.read_bytes()
        result.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return result


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_path(path: Path) -> tuple[str, int, str, str]:
    if path.is_file():
        payload = path.read_bytes()
        return (
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            "file",
            "sha256-exact-file-bytes/v1",
        )
    manifest = canonical_directory_manifest(path)
    return (
        hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        sum(int(item["size_bytes"]) for item in manifest),
        "directory",
        "sha256-canonical-json-manifest/v1",
    )


def record_artifact(
    path: Path, source: str, revision: str, acquired_at_utc: str
) -> ArtifactRecord:
    sha256, size_bytes, path_kind, identity_algorithm = hash_path(path)
    return ArtifactRecord(
        source=source,
        revision=revision,
        acquired_at_utc=acquired_at_utc,
        sha256=sha256,
        size_bytes=size_bytes,
        path_name=path.name,
        path_kind=path_kind,
        identity_algorithm=identity_algorithm,
    )


def register(
    registry_path: Path,
    kind: Literal["training_data", "cot_activation_interface"],
    record: ArtifactRecord,
) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry[kind] is not None:
        raise ArtifactAlreadyRegistered(f"{kind} already registered")
    registry[kind] = asdict(record)
    registry["gate_open"] = (
        registry["training_data"] is not None
        and registry["cot_activation_interface"] is not None
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry

