import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_three_lanes_and_execution_surfaces_are_explicit():
    state = (ROOT / "COMPETITION_STATE.md").read_text(encoding="utf-8")
    assert "YOLO EXPLORATORY COMPETITION" in state
    assert "LOCAL BASELINE REPRODUCTION:       ENVIRONMENT_BLOCKED" in state
    assert "EXTERNAL CODABENCH CONTRACT SMOKE: PASSED" in state
    assert "EXTERNAL SUBMISSION ID:            915072" in state
    assert re.search(r"SCIENTIFIC EXECUTION:\s+NOT AUTHORIZED", state)


def test_scientific_registry_remains_closed():
    registry = (ROOT / "RELEASE_REGISTRY.json").read_text(encoding="utf-8")
    assert '"training_data": null' in registry
    assert '"cot_activation_interface": null' in registry
    assert '"gate_open": false' in registry
