import json
from pathlib import Path

import pytest

from aimo_interp_yolo.protocol import load_protocol, member_by_id


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "yolo" / "batteries" / "YOLO001-B1.protocol.json"


def write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_b1_protocol_has_exact_frozen_matrix():
    protocol = load_protocol(PROTOCOL)
    assert protocol.schema == "aimo-interp-yolo-battery-protocol/v0.1"
    assert protocol.battery_id == "YOLO001-B1"
    assert protocol.protocol_state == "OPEN"
    assert protocol.track == "small"
    assert [member.protocol_id for member in protocol.members] == [
        "Y001-A", "Y001-B", "Y001-C", "Y001-D",
        "Y001-E", "Y001-F", "CTRL-T", "CTRL-F",
    ]
    assert member_by_id(protocol, "Y001-A").temperature_schedule == (0.7, 0.7, 0.7)
    assert member_by_id(protocol, "Y001-B").generation_count == 5
    assert member_by_id(protocol, "Y001-C").agreement_rule == "AT_LEAST_4_OF_5"
    assert member_by_id(protocol, "Y001-F").temperature_schedule == (0.4, 0.4, 0.9, 0.9)
    assert member_by_id(protocol, "CTRL-T").constant_prediction is True
    assert member_by_id(protocol, "CTRL-F").constant_prediction is False


def test_b1_protocol_has_exact_runtime_and_generation_call_contracts():
    protocol = load_protocol(PROTOCOL)
    assert protocol.runtime_contract["entrypoint"] == (
        "are_robust(model_id: str, problems: list[str]) -> list[bool]"
    )
    assert protocol.runtime_contract["model_id_routing"] == {
        "default": "DIRECT_SUPPLIED_MODEL_ID",
        "known_aliases": {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"},
        "unknown_alias_policy": "NO_SPECULATIVE_ALIAS_DIRECT_LOAD",
        "load_failure_action": "RAISE",
    }
    assert [member.generation_calls_per_problem for member in protocol.members] == [
        3, 5, 5, 3, 3, 4, 0, 0,
    ]


def test_protocol_is_result_blind_recursively(tmp_path: Path):
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["runtime_contract"]["nested"] = [{"score": 0.5}]
    path = tmp_path / "scored.json"
    write_payload(path, payload)
    with pytest.raises(ValueError, match="result-bearing key"):
        load_protocol(path)


def test_duplicate_member_id_is_rejected(tmp_path: Path):
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["members"][1]["protocol_id"] = "Y001-A"
    path = tmp_path / "bad.json"
    write_payload(path, payload)
    with pytest.raises(ValueError, match="duplicate protocol_id"):
        load_protocol(path)


def test_temperature_mutation_is_rejected(tmp_path: Path):
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["members"][0]["temperature_schedule"] = [0.8, 0.8, 0.8]
    path = tmp_path / "mutated.json"
    write_payload(path, payload)
    with pytest.raises(ValueError, match="frozen member mismatch"):
        load_protocol(path)


@pytest.mark.parametrize(
    "location",
    [
        ("runtime_contract",),
        ("runtime_contract", "model_id_routing"),
        ("shared_generation_contract",),
        ("members", 0),
    ],
)
def test_unknown_semantic_keys_are_rejected(tmp_path: Path, location: tuple[object, ...]):
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    target: object = payload
    for key in location:
        target = target[key]  # type: ignore[index]
    target["unexpected"] = "value"  # type: ignore[index]
    path = tmp_path / "unknown.json"
    write_payload(path, payload)
    with pytest.raises(ValueError, match="unexpected keys"):
        load_protocol(path)


def test_missing_nested_semantic_key_is_rejected(tmp_path: Path):
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    del payload["runtime_contract"]["network"]
    path = tmp_path / "missing.json"
    write_payload(path, payload)
    with pytest.raises(ValueError, match="missing keys"):
        load_protocol(path)
