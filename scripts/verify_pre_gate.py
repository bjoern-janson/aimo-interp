import importlib.util
import json
from pathlib import Path

from aimo_interp_infra.upstream import UpstreamLock

ROOT = Path(__file__).resolve().parents[1]


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    if lock.commit != "e98c489a98acb6c833588dca74228bee9782d5dd":
        raise SystemExit("unexpected upstream commit")

    registry = json.loads(
        (ROOT / "RELEASE_REGISTRY.json").read_text(encoding="utf-8")
    )
    if registry["gate_open"]:
        raise SystemExit(
            "external gate is open; pre-gate verifier must be replaced by "
            "the separately authorized observational-audit workflow"
        )

    forbidden = [
        ROOT / "solutions",
        ROOT / "features",
        ROOT / "models",
        ROOT / "PREREGISTRATION.md",
        ROOT / "SCIENTIFIC_RESULT.json",
    ]
    present = [path for path in forbidden if path.exists()]
    if present:
        raise SystemExit(f"scientific objects exist before gate: {present}")

    for control in ("all-true", "all-false"):
        path = ROOT / "controls" / control / "solution.py"
        module = _load_solution(path)
        function = getattr(module, "are_robust", None)
        if not callable(function):
            raise SystemExit(f"{path} has no callable are_robust")
        output = function("example/model", ["p1", "p2"])
        if len(output) != 2 or any(type(value) is not bool for value in output):
            raise SystemExit(f"{path} violates native-bool batch contract")

    print("PRE_GATE_INFRASTRUCTURE_VALID")
    print("SCIENTIFIC_EXECUTION_NOT_AUTHORIZED")


if __name__ == "__main__":
    main()

