from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from aimo_interp_infra.packaging import build_submission

from .jsonutil import canonical_json_bytes
from .protocol import BatteryProtocol, MemberProtocol, load_protocol

RUNTIME_FILES = (
    "__init__.py", "jsonutil.py", "protocol.py", "core.py",
    "model_runtime.py", "submission.py",
)
SOLUTION = '''from pathlib import Path
from aimo_interp_yolo.submission import are_robust_from_bundle
_MEMBER = Path(__file__).with_name("member.json")
def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return are_robust_from_bundle(_MEMBER, model_id, problems)
'''


@dataclass(frozen=True)
class BuiltArtifact:
    protocol_id: str
    zip_path: Path
    zip_sha256: str
    zip_size_bytes: int
    source_identity: str


def member_payload(protocol: BatteryProtocol, member: MemberProtocol) -> dict[str, object]:
    return {
        "schema": "aimo-interp-yolo-member/v0.1",
        "battery_id": protocol.battery_id,
        "member": asdict(member),
    }


def build_member_source(
    protocol: BatteryProtocol,
    member: MemberProtocol,
    staging_dir: Path,
    package_dir: Path,
) -> Path:
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)
    (staging_dir / "solution.py").write_text(SOLUTION, encoding="utf-8")
    (staging_dir / "member.json").write_bytes(canonical_json_bytes(member_payload(protocol, member)))
    runtime_dir = staging_dir / "aimo_interp_yolo"
    runtime_dir.mkdir()
    for filename in RUNTIME_FILES:
        shutil.copyfile(package_dir / filename, runtime_dir / filename)
    return staging_dir


def build_battery(
    protocol_path: Path,
    output_dir: Path,
    staging_root: Path,
    package_dir: Path,
) -> dict[str, BuiltArtifact]:
    protocol = load_protocol(protocol_path)
    built: dict[str, BuiltArtifact] = {}
    for member in protocol.members:
        source_dir = build_member_source(protocol, member, staging_root / member.protocol_id, package_dir)
        slug = member.protocol_id.casefold()
        destination = output_dir / f"yolo001-b1-{slug}-small.zip"
        zip_sha256 = build_submission(source_dir, destination, small=True)
        built[member.protocol_id] = BuiltArtifact(
            protocol_id=member.protocol_id,
            zip_path=destination,
            zip_sha256=zip_sha256,
            zip_size_bytes=destination.stat().st_size,
            source_identity=f"yolo/batteries/{member.protocol_id}",
        )
    return built
