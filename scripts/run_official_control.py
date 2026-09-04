import argparse
from pathlib import Path

from aimo_interp_infra.official_harness import run_official

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", choices=["all-true", "all-false"])
    parser.add_argument("--small", action="store_true")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / ".cache" / "getting-started" / "data" / "val-sample" / "input",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / ".cache" / "getting-started" / "data" / "val-sample" / "reference",
    )
    args = parser.parse_args()
    result = run_official(
        upstream_checkout=ROOT / ".cache" / "getting-started",
        solution=ROOT / "controls" / args.control,
        input_dir=args.input_dir,
        reference_dir=args.reference_dir,
        small=args.small,
    )
    print(
        {
            "accuracy": result.accuracy,
            "coverage": result.coverage,
            "invalid_predictions": result.invalid_predictions,
        }
    )


if __name__ == "__main__":
    main()

