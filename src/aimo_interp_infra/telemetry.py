from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

try:
    import resource
except ImportError:
    resource = None


@dataclass(frozen=True)
class TelemetryReceipt:
    schema: str
    label: str
    wall_seconds: float
    max_rss_bytes: int | None
    cuda_peak_allocated_bytes_by_device: dict[str, int] | None


def _max_rss_bytes() -> int | None:
    if resource is None:
        return None
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _load_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _reset_cuda_peak_memory_stats(torch_module: Any | None) -> None:
    if torch_module is None or not torch_module.cuda.is_available():
        return
    for device in range(torch_module.cuda.device_count()):
        torch_module.cuda.reset_peak_memory_stats(device)


def _cuda_peak_allocated_bytes_by_device(
    torch_module: Any | None,
) -> dict[str, int] | None:
    if torch_module is None or not torch_module.cuda.is_available():
        return None
    return {
        f"cuda:{device}": int(torch_module.cuda.max_memory_allocated(device))
        for device in range(torch_module.cuda.device_count())
    }


@contextmanager
def measure_runtime(label: str) -> Iterator[SimpleNamespace]:
    torch_module = _load_torch()
    _reset_cuda_peak_memory_stats(torch_module)
    started = time.perf_counter()
    holder = SimpleNamespace(receipt=None)
    try:
        yield holder
    finally:
        holder.receipt = TelemetryReceipt(
            schema="aimo-interp-telemetry/v0.2",
            label=label,
            wall_seconds=float(time.perf_counter() - started),
            max_rss_bytes=_max_rss_bytes(),
            cuda_peak_allocated_bytes_by_device=(
                _cuda_peak_allocated_bytes_by_device(torch_module)
            ),
        )


def write_receipt(receipt: TelemetryReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

