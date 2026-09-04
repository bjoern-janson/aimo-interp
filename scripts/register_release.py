import argparse
from datetime import datetime, timezone
from pathlib import Path

from aimo_interp_infra.release_gate import record_artifact, register

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind", choices=["training_data", "cot_activation_interface"]
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    acquired_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    record = record_artifact(
        args.artifact, args.source, args.revision, acquired_at.replace("+00:00", "Z")
    )
    registry = register(ROOT / "RELEASE_REGISTRY.json", args.kind, record)
    print(f"{args.kind}: sha256={record.sha256}")
    print(f"identity_algorithm={record.identity_algorithm}")
    print(f"gate_open={registry['gate_open']}")


if __name__ == "__main__":
    main()

