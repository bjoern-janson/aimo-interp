import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pre_gate_state_is_explicit_and_non_scientific():
    state = (ROOT / "COMPETITION_STATE.md").read_text(encoding="utf-8")
    assert "ENTER" in state
    assert "SMALL" in state
    assert "WAITING_FOR_EXTERNAL_GATE" in state
    assert re.search(r"SCIENTIFIC EXECUTION:\s+NOT AUTHORIZED", state)
    assert re.search(r"LABEL REPLAY:\s+CLOSED", state)


def test_upstream_lock_pins_exact_starter_commit():
    lock = json.loads((ROOT / "UPSTREAM_LOCK.json").read_text(encoding="utf-8"))
    assert lock["repository"] == "https://github.com/aimo-interp/getting-started.git"
    assert lock["commit"] == "e98c489a98acb6c833588dca74228bee9782d5dd"
    assert lock["default_branch"] == "main"
    assert lock["acquired_at_utc"] == "2026-09-04T12:11:48Z"
    assert "https://aimo-interp.github.io/" in lock["references"]
    assert lock["gate_artifact_registry"] == "RELEASE_REGISTRY.json"


def test_release_registry_starts_closed():
    registry = json.loads((ROOT / "RELEASE_REGISTRY.json").read_text(encoding="utf-8"))
    assert registry == {
        "schema": "aimo-interp-release-registry/v0.2",
        "training_data": None,
        "cot_activation_interface": None,
        "gate_open": False,
    }


def test_no_scientific_method_exists_before_gate():
    forbidden = [
        ROOT / "solutions",
        ROOT / "features",
        ROOT / "models",
        ROOT / "PREREGISTRATION.md",
        ROOT / "SCIENTIFIC_RESULT.json",
    ]
    assert [str(path) for path in forbidden if path.exists()] == []


def test_candidate_h0_is_not_marked_preregistered():
    state = (ROOT / "COMPETITION_STATE.md").read_text(encoding="utf-8")
    assert "SCIENTIFIC PREREGISTRATION: NOT AUTHORIZED" in state

