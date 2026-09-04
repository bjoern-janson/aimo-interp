from pathlib import Path

from aimo_interp_infra.upstream import UpstreamLock, materialize_checkout

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    destination = ROOT / ".cache" / "getting-started"
    materialize_checkout(lock, destination)
    print(destination)

