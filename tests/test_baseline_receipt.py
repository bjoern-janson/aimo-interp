import json
from pathlib import Path

import pytest

from aimo_interp_infra.baseline_receipt import (
    CONTRACT_SMOKE_PASS,
    BaselineReceipt,
    write_baseline_receipt,
)


def test_smoke_receipt_requires_complete_valid_output(tmp_path: Path):
    receipt = BaselineReceipt(
        schema="aimo-interp-baseline-receipt/v0.2",
        upstream_commit="e98c489a98acb6c833588dca74228bee9782d5dd",
        solution_path="solutions/trained-probe",
        status=CONTRACT_SMOKE_PASS,
        accuracy=0.5,
        coverage=1.0,
        invalid_predictions=0,
        wall_seconds=1.0,
        max_rss_bytes=None,
        cuda_peak_allocated_bytes_by_device=None,
    )
    output = tmp_path / "receipt.json"
    write_baseline_receipt(receipt, output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == CONTRACT_SMOKE_PASS


@pytest.mark.parametrize("coverage, invalid", [(0.9, 0), (1.0, 1)])
def test_smoke_rejects_incomplete_or_invalid_output(coverage: float, invalid: int):
    with pytest.raises(ValueError):
        BaselineReceipt(
            schema="aimo-interp-baseline-receipt/v0.2",
            upstream_commit="e98c489a98acb6c833588dca74228bee9782d5dd",
            solution_path="solutions/trained-probe",
            status=CONTRACT_SMOKE_PASS,
            accuracy=0.5,
            coverage=coverage,
            invalid_predictions=invalid,
            wall_seconds=1.0,
            max_rss_bytes=None,
            cuda_peak_allocated_bytes_by_device=None,
        )

