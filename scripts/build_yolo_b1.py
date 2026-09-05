import argparse
from pathlib import Path

from aimo_interp_yolo.build import build_battery

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "yolo/batteries/YOLO001-B1.protocol.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/yolo001-b1")
    parser.add_argument("--staging-root", type=Path, default=ROOT / ".build/yolo001-b1")
    args = parser.parse_args()
    built = build_battery(args.protocol, args.output_dir, args.staging_root, ROOT / "src/aimo_interp_yolo")
    for protocol_id, artifact in built.items():
        print(protocol_id, artifact.zip_sha256, artifact.zip_path)


if __name__ == "__main__":
    main()
