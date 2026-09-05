import argparse
from pathlib import Path

from aimo_interp_yolo.protocol import load_protocol
from aimo_interp_yolo.results import ResultEvent, append_result, derive_evaluation_state, read_events

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "yolo/batteries/YOLO001-B1.protocol.json")
    parser.add_argument("--custody", type=Path, default=ROOT / "yolo/batteries/YOLO001-B1.custody.json")
    parser.add_argument("--results", type=Path, default=ROOT / "yolo/results/YOLO001-B1.results.jsonl")
    for name in ("schema", "battery-id", "protocol-id", "zip-sha256", "closure-commit-sha", "codabench-submission-id",
                 "attempt-index", "submitted-at-utc", "observed-at-utc", "execution-status", "accuracy", "coverage",
                 "invalid-predictions", "rank-if-observed", "leaderboard-snapshot-context", "notes"):
        parser.add_argument(f"--{name}", required=name not in {"schema", "battery-id", "accuracy", "coverage", "invalid-predictions", "rank-if-observed", "leaderboard-snapshot-context", "notes"})
    args = parser.parse_args()
    def optional_float(value): return None if value is None else float(value)
    def optional_int(value): return None if value is None else int(value)
    event = ResultEvent(args.schema or "aimo-interp-yolo-result-event/v0.1", args.battery_id or "YOLO001-B1", args.protocol_id,
                        args.zip_sha256, args.closure_commit_sha, args.codabench_submission_id, int(args.attempt_index),
                        args.submitted_at_utc, args.observed_at_utc, args.execution_status, optional_float(args.accuracy),
                        optional_float(args.coverage), optional_int(args.invalid_predictions), optional_int(args.rank_if_observed),
                        args.leaderboard_snapshot_context, args.notes or "")
    append_result(args.protocol, args.custody, args.results, event)
    protocol = load_protocol(args.protocol)
    print(derive_evaluation_state([m.protocol_id for m in protocol.members], read_events(args.results)))


if __name__ == "__main__":
    main()
