import argparse
from pathlib import Path

from aimo_interp_yolo.custody import verify_committed_closure, verify_precommit_closure

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("precommit", "committed"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "yolo/batteries/YOLO001-B1.protocol.json")
    parser.add_argument("--custody", type=Path, default=ROOT / "yolo/batteries/YOLO001-B1.custody.json")
    parser.add_argument("--results", type=Path, default=ROOT / "yolo/results/YOLO001-B1.results.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/yolo001-b1")
    args = parser.parse_args()
    if args.mode == "precommit":
        verify_precommit_closure(args.protocol, args.custody, args.results, args.output_dir, ROOT)
    else:
        verify_committed_closure(args.protocol, args.custody, args.results, ROOT)
    print("B1_CLOSURE_VERIFIED")


if __name__ == "__main__":
    main()
