import json
from pathlib import Path

import aimo_interp_infra.telemetry as telemetry
from aimo_interp_infra.telemetry import (
    _cuda_peak_allocated_bytes_by_device,
    _max_rss_bytes,
    _reset_cuda_peak_memory_stats,
    measure_runtime,
    write_receipt,
)


class FakeCuda:
    def __init__(self) -> None:
        self.resets: list[int] = []

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def reset_peak_memory_stats(self, device: int) -> None:
        self.resets.append(device)

    def max_memory_allocated(self, device: int) -> int:
        return {0: 17, 1: 29}[device]


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()


def test_cuda_peaks_are_reset_and_recorded_per_device():
    torch_module = FakeTorch()
    _reset_cuda_peak_memory_stats(torch_module)
    assert torch_module.cuda.resets == [0, 1]
    assert _cuda_peak_allocated_bytes_by_device(torch_module) == {
        "cuda:0": 17,
        "cuda:1": 29,
    }


def test_missing_resource_returns_none(monkeypatch):
    monkeypatch.setattr(telemetry, "resource", None)
    assert _max_rss_bytes() is None


def test_receipt_is_nullable_and_json_serializable(tmp_path: Path):
    with measure_runtime("synthetic") as recorder:
        sum(range(100))

    receipt = recorder.receipt
    assert receipt.wall_seconds >= 0.0
    assert receipt.max_rss_bytes is None or receipt.max_rss_bytes >= 0

    path = tmp_path / "receipt.json"
    write_receipt(receipt, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "aimo-interp-telemetry/v0.2"

