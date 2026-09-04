from pathlib import Path

from aimo_interp_infra.upstream import UpstreamLock, verify_checkout

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    checkout = ROOT / ".cache" / "getting-started"
    verify_checkout(lock, checkout)
    print(f"verified {lock.github_repository}@{lock.commit}")

