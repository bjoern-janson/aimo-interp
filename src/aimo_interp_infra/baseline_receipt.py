from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONTRACT_SMOKE_PASS = "CONTRACT_SMOKE_PASS"


@dataclass(frozen=True)
class BaselineReceipt:
    schema: str
    upstream_commit: str
    solution_path: str
    status: str
    accuracy: float | None
    coverage: float | None
    invalid_predictions: int | None
    wall_seconds: float
    max_rss_bytes: int | None
    cuda_peak_allocated_bytes_by_device: dict[str, int] | None

    def __post_init__(self) -> None:
        if self.status != CONTRACT_SMOKE_PASS:
            raise ValueError("status must be CONTRACT_SMOKE_PASS")
        if self.coverage != 1.0:
            raise ValueError("CONTRACT_SMOKE_PASS requires coverage == 1.0")
        if self.invalid_predictions != 0:
            raise ValueError(
                "CONTRACT_SMOKE_PASS requires invalid_predictions == 0"
            )


def write_baseline_receipt(receipt: BaselineReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

