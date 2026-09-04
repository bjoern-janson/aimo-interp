import argparse
from pathlib import Path

from aimo_interp_infra.baseline_receipt import (
    CONTRACT_SMOKE_PASS,
    BaselineReceipt,
    write_baseline_receipt,
)
from aimo_interp_infra.official_harness import run_official
from aimo_interp_infra.telemetry import measure_runtime
from aimo_interp_infra.upstream import UpstreamLock, verify_checkout

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".cache" / "getting-started"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument(
        "--input-dir", type=Path, default=UPSTREAM / "data" / "val-sample" / "input"
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=UPSTREAM / "data" / "val-sample" / "reference"
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "receipts" / "official-baseline-contract-smoke.json",
    )
    args = parser.parse_args()

    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    verify_checkout(lock, UPSTREAM)
    with measure_runtime("official-baseline-contract-smoke") as telemetry:
        result = run_official(
            upstream_checkout=UPSTREAM,
            solution=UPSTREAM / "solutions" / "trained-probe",
            input_dir=args.input_dir,
            reference_dir=args.reference_dir,
            small=args.small,
        )
    runtime = telemetry.receipt
    write_baseline_receipt(
        BaselineReceipt(
            schema="aimo-interp-baseline-receipt/v0.2",
            upstream_commit=lock.commit,
            solution_path="solutions/trained-probe",
            status=CONTRACT_SMOKE_PASS,
            accuracy=result.accuracy,
            coverage=result.coverage,
            invalid_predictions=result.invalid_predictions,
            wall_seconds=runtime.wall_seconds,
            max_rss_bytes=runtime.max_rss_bytes,
            cuda_peak_allocated_bytes_by_device=runtime.cuda_peak_allocated_bytes_by_device,
        ),
        args.receipt,
    )
    print(args.receipt)


if __name__ == "__main__":
    main()

