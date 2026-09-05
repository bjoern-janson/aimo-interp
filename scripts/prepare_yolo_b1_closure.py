import argparse
from pathlib import Path

from aimo_interp_yolo.custody import prepare_closure

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--closed-at-utc", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "yolo/batteries/YOLO001-B1.protocol.json")
    parser.add_argument("--custody", type=Path, default=ROOT / "yolo/batteries/YOLO001-B1.custody.json")
    parser.add_argument("--results", type=Path, default=ROOT / "yolo/results/YOLO001-B1.results.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/yolo001-b1")
    parser.add_argument("--staging-root", type=Path, default=ROOT / ".build/yolo001-b1")
    args = parser.parse_args()
    prepare_closure(args.protocol, args.custody, args.results, args.output_dir, args.staging_root,
                    ROOT / "src/aimo_interp_yolo", ROOT, args.implementation_commit, args.closed_at_utc)
    print("B1_CLOSURE_PREPARED")


if __name__ == "__main__":
    main()
