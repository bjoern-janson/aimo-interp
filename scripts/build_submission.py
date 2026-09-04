import argparse
from pathlib import Path

from aimo_interp_infra.packaging import build_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    digest = build_submission(args.source, args.destination, args.small)
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()

