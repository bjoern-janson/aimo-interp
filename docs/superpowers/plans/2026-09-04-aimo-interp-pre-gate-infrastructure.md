# AIMO Interpretability Pre-Gate Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, provenance-preserving AIMO Interpretability competition infrastructure lane that reproduces the official execution contract, validates packaging/model lifecycle/telemetry, and stops before any scientific feature selection or robustness-model training.

**Architecture:** The repository will keep the official AIMO starter as an external, pinned authority rather than copying and silently modifying it. A small Python package will provide lock verification, an official-harness subprocess bridge, deterministic ZIP construction, generic batch model lifecycle/telemetry, and release-gate custody records. Scientific methods remain absent until the official training data and CoT activation interface are both released and audited.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, Python standard library; optional runtime hooks for PyTorch when available; official competition behavior is delegated to `aimo-interp/getting-started@e98c489a98acb6c833588dca74228bee9782d5dd`.

**Spec:** `docs/superpowers/specs/2026-09-04-aimo-interp-design.md`

## Global Constraints

- Repository: `bjoern-janson/aimo-interp`.
- Primary track: Small Models Track; Main Track is secondary.
- Pinned upstream starter: `aimo-interp/getting-started@e98c489a98acb6c833588dca74228bee9782d5dd`.
- Competition entry point remains exactly `are_robust(model_id: str, problems: list[str]) -> list[bool]`.
- Returned predictions must be native Python `bool`, preserve input order, and have exactly one value per problem.
- Evaluation is offline; implementation must not require inference-time network access.
- Do not choose or tune trajectory features against robustness labels before the external gate.
- Do not train a competition robustness classifier before the external gate except by running the unmodified official baselines as execution checks.
- Do not reconstruct hidden labels through runtime perturbation replay.
- Do not use leaderboard feedback to choose feature families, hypotheses, thresholds, or model classes.
- Do not reopen Γ, L2-v1, Reach, or any closed scientific lineage.
- The scientific gate remains `TRAINING DATA + CoT ACTIVATION INTERFACE`.
- The candidate future H0 is recorded only as non-executable context; no preregistration occurs in this plan.
- Infrastructure failures must fail loudly locally; they must not silently become robustness predictions.
- The public/warmup sample is a contract check, not an optimization oracle.
- The implementation phase ends in `WAITING_FOR_EXTERNAL_GATE`; no scientific experiment is opened.

---

## File Structure Locked by This Plan

```text
README.md
pyproject.toml
UPSTREAM_LOCK.json
COMPETITION_STATE.md
RESEARCH_LEDGER.md
RELEASE_REGISTRY.json

.github/
  workflows/
    ci.yml

src/
  aimo_interp_infra/
    __init__.py
    upstream.py
    official_harness.py
    packaging.py
    model_lifecycle.py
    telemetry.py
    release_gate.py
    baseline_receipt.py

scripts/
  materialize_upstream.py
  verify_upstream.py
  run_official_control.py
  build_submission.py
  register_release.py
  verify_pre_gate.py
  reproduce_official_baseline.py

controls/
  all-true/
    solution.py
  all-false/
    solution.py

tests/
  test_upstream.py
  test_official_harness.py
  test_packaging.py
  test_model_lifecycle.py
  test_telemetry.py
  test_release_gate.py
  test_baseline_receipt.py
  test_pre_gate_state.py

vendor/
  README.md

runtime/
  README.md

receipts/
  README.md

docs/
  superpowers/
    specs/
      2026-09-04-aimo-interp-design.md
    plans/
      2026-09-04-aimo-interp-pre-gate-infrastructure.md
```

`docs/superpowers/specs/2026-09-04-aimo-interp-design.md` already exists and is immutable except through a separately approved design revision.

---

### Task 1: Establish the repository constitution and Python package skeleton

**Files:**
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `UPSTREAM_LOCK.json`
- Create: `COMPETITION_STATE.md`
- Create: `RESEARCH_LEDGER.md`
- Create: `RELEASE_REGISTRY.json`
- Create: `src/aimo_interp_infra/__init__.py`
- Create: `vendor/README.md`
- Create: `runtime/README.md`
- Test: `tests/test_pre_gate_state.py`

**Interfaces:**
- Consumes: approved design spec.
- Produces: canonical repository state files and package import `aimo_interp_infra`.

- [ ] **Step 1: Write the failing state test**

Create `tests/test_pre_gate_state.py`:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pre_gate_state_is_explicit_and_non_scientific():
    state = (ROOT / "COMPETITION_STATE.md").read_text(encoding="utf-8")
    assert "ENTER" in state
    assert "SMALL" in state
    assert "WAITING_FOR_EXTERNAL_GATE" in state
    assert "SCIENTIFIC EXECUTION: NOT AUTHORIZED" in state
    assert "LABEL REPLAY: CLOSED" in state


def test_upstream_lock_pins_exact_starter_commit():
    lock = json.loads((ROOT / "UPSTREAM_LOCK.json").read_text(encoding="utf-8"))
    assert lock["repository"] == "https://github.com/aimo-interp/getting-started.git"
    assert lock["commit"] == "e98c489a98acb6c833588dca74228bee9782d5dd"
    assert lock["default_branch"] == "main"
    assert lock["acquired_at_utc"] == "2026-09-04T12:11:48Z"
    assert "https://aimo-interp.github.io/" in lock["references"]


def test_release_registry_starts_closed():
    registry = json.loads((ROOT / "RELEASE_REGISTRY.json").read_text(encoding="utf-8"))
    assert registry == {
        "schema": "aimo-interp-release-registry/v0.1",
        "training_data": None,
        "cot_activation_interface": None,
        "gate_open": False,
    }
```

- [ ] **Step 2: Run the state test and verify it fails**

Run:

```bash
uv run pytest tests/test_pre_gate_state.py -v
```

Expected: FAIL because the repository state files and package skeleton do not exist yet.

- [ ] **Step 3: Add the minimal project configuration and state files**

Create `pyproject.toml`:

```toml
[project]
name = "aimo-interp-infra"
version = "0.1.0"
description = "Pre-gate infrastructure for the AIMO Interpretability Challenge."
requires-python = ">=3.11"
dependencies = []

[dependency-groups]
dev = [
  "pytest>=8.0,<9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/aimo_interp_infra"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `src/aimo_interp_infra/__init__.py`:

```python
"""Infrastructure-only tooling for the AIMO Interpretability Challenge."""

__all__ = []
```

Create `UPSTREAM_LOCK.json`:

```json
{
  "schema": "aimo-interp-upstream-lock/v0.1",
  "repository": "https://github.com/aimo-interp/getting-started.git",
  "github_repository": "aimo-interp/getting-started",
  "default_branch": "main",
  "commit": "e98c489a98acb6c833588dca74228bee9782d5dd",
  "commit_date_utc": "2026-09-02T11:43:03Z",
  "commit_message": "fix: use shared leaderboard metric keys",
  "acquired_at_utc": "2026-09-04T12:11:48Z",
  "references": [
    "https://aimo-interp.github.io/",
    "https://arxiv.org/abs/2607.13899",
    "https://github.com/aimo-interp/getting-started"
  ],
  "authority": "PINNED_EXECUTION_CONTRACT"
}
```

Create `RELEASE_REGISTRY.json`:

```json
{
  "schema": "aimo-interp-release-registry/v0.1",
  "training_data": null,
  "cot_activation_interface": null,
  "gate_open": false
}
```

Create `COMPETITION_STATE.md` with this exact state block:

````markdown
# Competition State

```text
COMPETITION DECISION:       ENTER
PRIMARY TRACK:              SMALL
SECONDARY TRACK:            MAIN
PHASE:                      WAITING_FOR_EXTERNAL_GATE

TRAINING DATA:              NOT REGISTERED
CoT ACTIVATION INTERFACE:   NOT REGISTERED
OBSERVATIONAL AUDIT:        NOT OPENED
SCIENTIFIC PREREGISTRATION: NOT AUTHORIZED
SCIENTIFIC EXECUTION:       NOT AUTHORIZED

LABEL REPLAY:               CLOSED
LEADERBOARD OPTIMIZATION:   NOT AUTHORIZED
FEATURE ONTOLOGY:           NOT FROZEN
```

The gate opens only after both official training data and the official CoT
activation interface have been released, registered with immutable provenance,
and passed a separate observational audit.

The current repository is infrastructure only.

## Contract authority layers

1. **Historical design:** `arXiv:2607.13899`.
2. **Pinned execution contract:** `aimo-interp/getting-started@e98c489a98acb6c833588dca74228bee9782d5dd`.
3. **Live organizer clarifications:** may narrow or supersede historical intent,
   but every change must be recorded before implementation absorbs it.

## Unresolved live-contract items

- exact final model inventory;
- exact live robustness-label aggregation rule;
- official training-data revision and schema;
- official CoT-activation interface revision and schema.

No unresolved item may be guessed into the implementation.
````

Create `RESEARCH_LEDGER.md`:

````markdown
# Research Ledger

## 2026-09-04 — Design freeze

**Question:** Can the repository prepare a competition-compatible execution lane
without spending scientific degrees of freedom before the observational universe
is known?

**Observation:** The infrastructure-first design was approved and committed as
the first repository object.

**Status:** `INFRASTRUCTURE_AUTHORIZED / SCIENCE_NOT_OPENED`

**Claim ceiling:** No robustness-predictive feature, classifier, causal
interpretation, or trajectory hypothesis is established.

**Next legal action:** Implement and verify infrastructure only, then wait for
the official training-data and CoT-activation releases.
````

Create `README.md`:

````markdown
# AIMO Interpretability — Pre-Gate Infrastructure

This repository is the competition execution lane for the AIMO Interpretability
Challenge.

Current authority:

```text
ENTER -> SMALL -> WAIT FOR TRAINING DATA + CoT ACTIVATIONS
      -> OBSERVATIONAL AUDIT -> PREREGISTRATION -> EXECUTION
```

No scientific robustness experiment is open.

The repository currently permits only provenance, official-contract
compatibility, deterministic packaging, runtime/model-lifecycle validation,
telemetry, and release-gate custody.

See:
- `docs/superpowers/specs/2026-09-04-aimo-interp-design.md`
- `COMPETITION_STATE.md`
- `RESEARCH_LEDGER.md`
- `UPSTREAM_LOCK.json`
````

Create `vendor/README.md`:

````markdown
# Vendor Boundary

The official `aimo-interp/getting-started` repository is not copied into this
tree. Development tooling materializes the exact revision recorded in
`UPSTREAM_LOCK.json` into a disposable local checkout.

No upstream update is absorbed implicitly.
````

Create `runtime/README.md`:

````markdown
# Runtime Boundary

Runtime code in this repository wraps the official Codabench contract but does
not redefine its label semantics.

The competition entry point is:

```python
are_robust(model_id: str, problems: list[str]) -> list[bool]
```

Infrastructure code may validate loading, ordering, coverage, packaging,
telemetry, and failure behavior. Scientific feature extraction is not
authorized before the external gate.
````

- [ ] **Step 4: Install dev dependencies and run the state test**

Run:

```bash
uv sync
uv run pytest tests/test_pre_gate_state.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the repository constitution**

```bash
git add README.md pyproject.toml UPSTREAM_LOCK.json COMPETITION_STATE.md \
  RESEARCH_LEDGER.md RELEASE_REGISTRY.json src/aimo_interp_infra/__init__.py \
  vendor/README.md runtime/README.md tests/test_pre_gate_state.py uv.lock
git commit -m "chore: establish AIMO pre-gate repository state"
```

---

### Task 2: Implement exact upstream lock verification and materialization

**Files:**
- Create: `src/aimo_interp_infra/upstream.py`
- Create: `scripts/materialize_upstream.py`
- Create: `scripts/verify_upstream.py`
- Test: `tests/test_upstream.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `UPSTREAM_LOCK.json`.
- Produces:
  - `UpstreamLock.from_json(path: Path) -> UpstreamLock`
  - `verify_checkout(lock: UpstreamLock, checkout: Path) -> None`
  - `materialize_checkout(lock: UpstreamLock, destination: Path) -> None`

- [ ] **Step 1: Write failing upstream-verification tests**

Create `tests/test_upstream.py`:

```python
import json
import subprocess
from pathlib import Path

import pytest

from aimo_interp_infra.upstream import (
    UpstreamLock,
    UpstreamMismatch,
    verify_checkout,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_lock_parses_exact_commit(tmp_path: Path):
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "repository": "https://example.test/repo.git",
                "github_repository": "owner/repo",
                "default_branch": "main",
                "commit": "a" * 40,
                "authority": "PINNED_EXECUTION_CONTRACT",
                "acquired_at_utc": "2026-09-04T12:00:00Z",
                "references": ["https://example.test/repo"],
            }
        ),
        encoding="utf-8",
    )
    lock = UpstreamLock.from_json(path)
    assert lock.commit == "a" * 40
    assert lock.github_repository == "owner/repo"


def test_verify_checkout_accepts_matching_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    head = _git(repo, "rev-parse", "HEAD")

    lock = UpstreamLock(
        repository="https://example.test/repo.git",
        github_repository="owner/repo",
        default_branch="main",
        commit=head,
        authority="PINNED_EXECUTION_CONTRACT",
    )
    verify_checkout(lock, repo)


def test_verify_checkout_rejects_wrong_head(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")

    lock = UpstreamLock(
        repository="https://example.test/repo.git",
        github_repository="owner/repo",
        default_branch="main",
        commit="b" * 40,
        authority="PINNED_EXECUTION_CONTRACT",
    )
    with pytest.raises(UpstreamMismatch):
        verify_checkout(lock, repo)
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
uv run pytest tests/test_upstream.py -v
```

Expected: FAIL because `aimo_interp_infra.upstream` does not exist.

- [ ] **Step 3: Implement the lock model and exact-head verifier**

Create `src/aimo_interp_infra/upstream.py`:

```python
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class UpstreamMismatch(RuntimeError):
    """Raised when a local upstream checkout does not match the frozen lock."""


@dataclass(frozen=True)
class UpstreamLock:
    repository: str
    github_repository: str
    default_branch: str
    commit: str
    authority: str
    acquired_at_utc: str | None = None
    references: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, path: Path) -> "UpstreamLock":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            repository=str(payload["repository"]),
            github_repository=str(payload["github_repository"]),
            default_branch=str(payload["default_branch"]),
            commit=str(payload["commit"]),
            authority=str(payload["authority"]),
            acquired_at_utc=str(payload["acquired_at_utc"]),
            references=tuple(str(item) for item in payload["references"]),
        )


def _run_git(*args: str, cwd: Path | None = None) -> str:
    command = ["git"]
    if cwd is not None:
        command += ["-C", str(cwd)]
    command += list(args)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def verify_checkout(lock: UpstreamLock, checkout: Path) -> None:
    head = _run_git("rev-parse", "HEAD", cwd=checkout)
    if head != lock.commit:
        raise UpstreamMismatch(
            f"upstream HEAD {head} does not match frozen commit {lock.commit}"
        )
    dirty = _run_git("status", "--porcelain", cwd=checkout)
    if dirty:
        raise UpstreamMismatch("upstream checkout contains uncommitted changes")


def materialize_checkout(lock: UpstreamLock, destination: Path) -> None:
    if destination.exists():
        verify_checkout(lock, destination)
        return

    if shutil.which("git") is None:
        raise RuntimeError("git executable is required to materialize upstream")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_git("clone", "--filter=blob:none", lock.repository, str(destination))
    _run_git("checkout", "--detach", lock.commit, cwd=destination)
    verify_checkout(lock, destination)
```

- [ ] **Step 4: Add exact CLI wrappers**

Create `scripts/materialize_upstream.py`:

```python
from pathlib import Path

from aimo_interp_infra.upstream import UpstreamLock, materialize_checkout

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    destination = ROOT / ".cache" / "getting-started"
    materialize_checkout(lock, destination)
    print(destination)
```

Create `scripts/verify_upstream.py`:

```python
from pathlib import Path

from aimo_interp_infra.upstream import UpstreamLock, verify_checkout

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    checkout = ROOT / ".cache" / "getting-started"
    verify_checkout(lock, checkout)
    print(f"verified {lock.github_repository}@{lock.commit}")
```

Append to `README.md`:

````markdown
## Materialize the frozen official starter

```bash
uv run scripts/materialize_upstream.py
uv run scripts/verify_upstream.py
```

The checkout lives under `.cache/getting-started` and must remain exactly at the
commit recorded in `UPSTREAM_LOCK.json`.
````

Add `.cache/` to `.gitignore`.

- [ ] **Step 5: Run the unit tests**

Run:

```bash
uv run pytest tests/test_upstream.py -v
```

Expected: PASS.

- [ ] **Step 6: Materialize and verify the real pinned upstream**

Run:

```bash
uv run scripts/materialize_upstream.py
uv run scripts/verify_upstream.py
git -C .cache/getting-started rev-parse HEAD
```

Expected final line:

```text
e98c489a98acb6c833588dca74228bee9782d5dd
```

- [ ] **Step 7: Commit upstream custody tooling**

```bash
git add .gitignore README.md src/aimo_interp_infra/upstream.py \
  scripts/materialize_upstream.py scripts/verify_upstream.py tests/test_upstream.py
git commit -m "feat: verify frozen AIMO upstream contract"
```

---

### Task 3: Bridge to the official ingestion/scoring harness without reimplementing it

**Files:**
- Create: `src/aimo_interp_infra/official_harness.py`
- Create: `scripts/run_official_control.py`
- Create: `controls/all-true/solution.py`
- Create: `controls/all-false/solution.py`
- Test: `tests/test_official_harness.py`
- Modify: `runtime/README.md`

**Interfaces:**
- Consumes: verified upstream checkout and a solution directory/ZIP.
- Produces:
  - `OfficialRunResult(accuracy, coverage, invalid_predictions, stdout, stderr)`
  - `run_official(upstream_checkout, solution, input_dir, reference_dir, small=False)`

- [ ] **Step 1: Write failing harness tests using a fake upstream runner**

Create `tests/test_official_harness.py`:

```python
from pathlib import Path

from aimo_interp_infra.official_harness import run_official


def test_run_official_parses_metrics_from_stdout(tmp_path: Path):
    upstream = tmp_path / "upstream"
    scripts = upstream / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_local.py"
    runner.write_text(
        "import json\n"
        "print(json.dumps({'accuracy': 0.5, 'coverage': 1.0, "
        "'invalid_predictions': 0}))\n",
        encoding="utf-8",
    )

    solution = tmp_path / "solution"
    solution.mkdir()
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    input_dir.mkdir()
    reference_dir.mkdir()

    result = run_official(
        upstream_checkout=upstream,
        solution=solution,
        input_dir=input_dir,
        reference_dir=reference_dir,
        small=False,
        uv_executable="uv",
    )

    assert result.accuracy == 0.5
    assert result.coverage == 1.0
    assert result.invalid_predictions == 0


def test_run_official_propagates_runner_failure(tmp_path: Path):
    upstream = tmp_path / "upstream"
    scripts = upstream / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_local.py"
    runner.write_text(
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )

    solution = tmp_path / "solution"
    solution.mkdir()
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    input_dir.mkdir()
    reference_dir.mkdir()

    import subprocess

    try:
        run_official(
            upstream_checkout=upstream,
            solution=solution,
            input_dir=input_dir,
            reference_dir=reference_dir,
            small=False,
            uv_executable="uv",
        )
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 7
    else:
        raise AssertionError("official runner failure must propagate")


def test_run_official_forwards_small_flag(tmp_path: Path):
    upstream = tmp_path / "upstream"
    scripts = upstream / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_local.py"
    runner.write_text(
        "import json, sys\n"
        "assert '--small' in sys.argv\n"
        "print(json.dumps({'accuracy': 1.0, 'coverage': 1.0, "
        "'invalid_predictions': 0}))\n",
        encoding="utf-8",
    )

    solution = tmp_path / "solution"
    solution.mkdir()
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    input_dir.mkdir()
    reference_dir.mkdir()

    result = run_official(
        upstream_checkout=upstream,
        solution=solution,
        input_dir=input_dir,
        reference_dir=reference_dir,
        small=True,
        uv_executable="uv",
    )
    assert result.accuracy == 1.0
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```bash
uv run pytest tests/test_official_harness.py -v
```

Expected: FAIL because `aimo_interp_infra.official_harness` does not exist.

- [ ] **Step 3: Implement the subprocess bridge**

Create `src/aimo_interp_infra/official_harness.py`:

```python
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialRunResult:
    accuracy: float
    coverage: float
    invalid_predictions: int
    stdout: str
    stderr: str


def _extract_metrics(stdout: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for start in range(len(stdout)):
        if stdout[start] != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if {"accuracy", "coverage", "invalid_predictions"} <= payload.keys():
            return payload
    raise RuntimeError("official runner output contained no score object")


def run_official(
    upstream_checkout: Path,
    solution: Path,
    input_dir: Path,
    reference_dir: Path,
    small: bool,
    uv_executable: str = "uv",
) -> OfficialRunResult:
    command = [
        uv_executable,
        "run",
        str(upstream_checkout / "scripts" / "run_local.py"),
        str(solution),
        "--input-dir",
        str(input_dir),
        "--reference-dir",
        str(reference_dir),
    ]
    if small:
        command.append("--small")

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        cwd=upstream_checkout,
    )
    metrics = _extract_metrics(completed.stdout)
    return OfficialRunResult(
        accuracy=float(metrics["accuracy"]),
        coverage=float(metrics["coverage"]),
        invalid_predictions=int(metrics["invalid_predictions"]),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
```

- [ ] **Step 4: Add the two allowed trivial controls**

Create `controls/all-true/solution.py`:

```python
def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return [True for _ in problems]
```

Create `controls/all-false/solution.py`:

```python
def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return [False for _ in problems]
```

Create `scripts/run_official_control.py`:

```python
import argparse
from pathlib import Path

from aimo_interp_infra.official_harness import run_official

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", choices=["all-true", "all-false"])
    parser.add_argument("--small", action="store_true")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / ".cache" / "getting-started" / "data" / "val-sample" / "input",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / ".cache" / "getting-started" / "data" / "val-sample" / "reference",
    )
    args = parser.parse_args()

    result = run_official(
        upstream_checkout=ROOT / ".cache" / "getting-started",
        solution=ROOT / "controls" / args.control,
        input_dir=args.input_dir,
        reference_dir=args.reference_dir,
        small=args.small,
    )
    print(
        {
            "accuracy": result.accuracy,
            "coverage": result.coverage,
            "invalid_predictions": result.invalid_predictions,
        }
    )


if __name__ == "__main__":
    main()
```

Append to `runtime/README.md`:

````markdown
The local score bridge invokes the pinned upstream `scripts/run_local.py`
directly. It parses the official score object but does not reimplement ingestion
or scoring.

Allowed pre-gate controls:
- `controls/all-true`
- `controls/all-false`
````

- [ ] **Step 5: Run the harness tests**

Run:

```bash
uv run pytest tests/test_official_harness.py -v
```

Expected: PASS.

- [ ] **Step 6: Exercise the real upstream control path**

Run:

```bash
cd .cache/getting-started
uv run scripts/import_hf_dataset.py
cd ../..

uv run scripts/run_official_control.py all-true
uv run scripts/run_official_control.py all-false
```

Expected for both runs:
- `coverage == 1.0`
- `invalid_predictions == 0`
- accuracy may differ and is recorded only as a contract check.

- [ ] **Step 7: Commit the official-harness bridge**

```bash
git add src/aimo_interp_infra/official_harness.py scripts/run_official_control.py \
  controls/all-true/solution.py controls/all-false/solution.py \
  tests/test_official_harness.py runtime/README.md
git commit -m "feat: bridge pinned official AIMO harness"
```

---

### Task 4: Build byte-for-byte deterministic competition ZIPs

**Files:**
- Create: `src/aimo_interp_infra/packaging.py`
- Create: `scripts/build_submission.py`
- Test: `tests/test_packaging.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: a submission source directory containing root `solution.py`.
- Produces:
  - `build_submission(source_dir: Path, destination: Path, small: bool) -> str`
  - returned string is SHA-256 of the ZIP bytes.

- [ ] **Step 1: Write failing deterministic-packaging tests**

Create `tests/test_packaging.py`:

```python
import hashlib
import zipfile
from pathlib import Path

from aimo_interp_infra.packaging import build_submission


def test_build_submission_is_byte_reproducible(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "solution.py").write_text(
        "def are_robust(model_id, problems):\n"
        "    return [False for _ in problems]\n",
        encoding="utf-8",
    )
    helper = source / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    hash1 = build_submission(source, first, small=False)
    hash2 = build_submission(source, second, small=False)

    assert first.read_bytes() == second.read_bytes()
    assert hash1 == hash2 == hashlib.sha256(first.read_bytes()).hexdigest()


def test_small_track_adds_root_marker_without_mutating_source(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "solution.py").write_text(
        "def are_robust(model_id, problems):\n"
        "    return [True for _ in problems]\n",
        encoding="utf-8",
    )
    destination = tmp_path / "small.zip"

    build_submission(source, destination, small=True)

    assert not (source / "small.txt").exists()
    with zipfile.ZipFile(destination) as archive:
        assert "solution.py" in archive.namelist()
        assert "small.txt" in archive.namelist()
        assert archive.read("small.txt") == b""


def test_build_rejects_missing_solution(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "bad.zip"

    try:
        build_submission(source, destination, small=False)
    except ValueError as exc:
        assert "solution.py" in str(exc)
    else:
        raise AssertionError("missing solution.py must be rejected")
```

- [ ] **Step 2: Run tests and verify the packaging module is missing**

Run:

```bash
uv run pytest tests/test_packaging.py -v
```

Expected: FAIL because `aimo_interp_infra.packaging` does not exist.

- [ ] **Step 3: Implement deterministic ZIP construction**

Create `src/aimo_interp_infra/packaging.py`:

```python
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _eligible_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source_dir).as_posix())


def _write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def build_submission(source_dir: Path, destination: Path, small: bool) -> str:
    solution = source_dir / "solution.py"
    if not solution.is_file():
        raise ValueError("submission source must contain root solution.py")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for path in _eligible_files(source_dir):
            relative = path.relative_to(source_dir).as_posix()
            if relative.casefold() == "small.txt":
                continue
            _write_bytes(archive, relative, path.read_bytes())
        if small:
            _write_bytes(archive, "small.txt", b"")

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return digest
```

- [ ] **Step 4: Add the build CLI**

Create `scripts/build_submission.py`:

```python
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
```

Append to `README.md`:

````markdown
## Deterministic packaging

```bash
uv run scripts/build_submission.py controls/all-false dist/all-false-small.zip --small
```

Rebuilding an unchanged source tree must reproduce the ZIP byte-for-byte.
````

- [ ] **Step 5: Run packaging tests**

Run:

```bash
uv run pytest tests/test_packaging.py -v
```

Expected: PASS.

- [ ] **Step 6: Build the same control twice and verify identical hashes**

Run:

```bash
mkdir -p dist
uv run scripts/build_submission.py controls/all-false dist/a.zip --small
uv run scripts/build_submission.py controls/all-false dist/b.zip --small
sha256sum dist/a.zip dist/b.zip
cmp dist/a.zip dist/b.zip
```

Expected: identical SHA-256 values and `cmp` exits 0.

- [ ] **Step 7: Commit deterministic packaging**

```bash
git add src/aimo_interp_infra/packaging.py scripts/build_submission.py \
  tests/test_packaging.py README.md
git commit -m "feat: add deterministic AIMO submission packaging"
```

---

### Task 5: Implement generic one-load-per-batch model lifecycle without scientific features

**Files:**
- Create: `src/aimo_interp_infra/model_lifecycle.py`
- Test: `tests/test_model_lifecycle.py`
- Modify: `runtime/README.md`

**Interfaces:**
- Consumes:
  - `loader(model_id: str) -> ModelT`
  - `processor(model: ModelT, problem: str) -> ResultT`
  - optional `releaser(model: ModelT) -> None`
- Produces:
  - `BatchModelExecutor.run(model_id: str, problems: list[str]) -> list[ResultT]`
- This is generic execution plumbing only; it contains no robustness feature or decision rule.

- [ ] **Step 1: Write failing lifecycle tests**

Create `tests/test_model_lifecycle.py`:

```python
from aimo_interp_infra.model_lifecycle import BatchModelExecutor


def test_executor_loads_exactly_once_for_one_batch():
    loads: list[str] = []
    releases: list[str] = []

    def loader(model_id: str) -> dict[str, str]:
        loads.append(model_id)
        return {"model_id": model_id}

    def processor(model: dict[str, str], problem: str) -> str:
        return f"{model['model_id']}::{problem}"

    def releaser(model: dict[str, str]) -> None:
        releases.append(model["model_id"])

    executor = BatchModelExecutor(loader, processor, releaser)
    output = executor.run("example/model", ["p1", "p2", "p3"])

    assert output == [
        "example/model::p1",
        "example/model::p2",
        "example/model::p3",
    ]
    assert loads == ["example/model"]
    assert releases == ["example/model"]


def test_executor_releases_model_when_processor_raises():
    released: list[str] = []

    def loader(model_id: str) -> dict[str, str]:
        return {"model_id": model_id}

    def processor(model: dict[str, str], problem: str) -> str:
        raise RuntimeError("synthetic processor failure")

    def releaser(model: dict[str, str]) -> None:
        released.append(model["model_id"])

    executor = BatchModelExecutor(loader, processor, releaser)

    try:
        executor.run("example/model", ["p1"])
    except RuntimeError as exc:
        assert "synthetic processor failure" in str(exc)
    else:
        raise AssertionError("processor failure must propagate")

    assert released == ["example/model"]
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```bash
uv run pytest tests/test_model_lifecycle.py -v
```

Expected: FAIL because `aimo_interp_infra.model_lifecycle` does not exist.

- [ ] **Step 3: Implement the generic batch executor**

Create `src/aimo_interp_infra/model_lifecycle.py`:

```python
from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

ModelT = TypeVar("ModelT")
ResultT = TypeVar("ResultT")


@dataclass
class BatchModelExecutor(Generic[ModelT, ResultT]):
    loader: Callable[[str], ModelT]
    processor: Callable[[ModelT, str], ResultT]
    releaser: Callable[[ModelT], None]

    def run(self, model_id: str, problems: list[str]) -> list[ResultT]:
        model = self.loader(model_id)
        try:
            return [self.processor(model, problem) for problem in problems]
        finally:
            self.releaser(model)
            del model
            gc.collect()
```

Append to `runtime/README.md`:

````markdown
`BatchModelExecutor` is a generic lifecycle primitive. It guarantees one loader
call and one releaser call per model batch. It intentionally has no
robustness-specific processor, feature extractor, threshold, or fallback.
````

- [ ] **Step 4: Run lifecycle tests**

Run:

```bash
uv run pytest tests/test_model_lifecycle.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit model-lifecycle plumbing**

```bash
git add src/aimo_interp_infra/model_lifecycle.py \
  tests/test_model_lifecycle.py runtime/README.md
git commit -m "feat: add one-load batch model lifecycle"
```

---

### Task 6: Add runtime and memory telemetry as non-scientific receipts

**Files:**
- Create: `src/aimo_interp_infra/telemetry.py`
- Test: `tests/test_telemetry.py`
- Modify: `runtime/README.md`

**Interfaces:**
- Produces:
  - `TelemetryReceipt`
  - `measure_runtime(label: str)` context manager
  - `write_receipt(receipt: TelemetryReceipt, path: Path) -> None`
- Telemetry may observe wall-clock time, process max RSS, and CUDA peak bytes when PyTorch/CUDA is available.

- [ ] **Step 1: Write failing telemetry tests**

Create `tests/test_telemetry.py`:

```python
import json
from pathlib import Path

from aimo_interp_infra.telemetry import measure_runtime, write_receipt


def test_measure_runtime_records_nonnegative_duration():
    with measure_runtime("synthetic") as recorder:
        sum(range(100))

    receipt = recorder.receipt
    assert receipt is not None
    assert receipt.label == "synthetic"
    assert receipt.wall_seconds >= 0.0
    assert receipt.max_rss_kib >= 0


def test_write_receipt_is_json_serializable(tmp_path: Path):
    with measure_runtime("synthetic") as recorder:
        pass

    path = tmp_path / "receipt.json"
    write_receipt(recorder.receipt, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "aimo-interp-telemetry/v0.1"
    assert payload["label"] == "synthetic"
    assert isinstance(payload["wall_seconds"], float)
```

- [ ] **Step 2: Run tests and verify the telemetry module is missing**

Run:

```bash
uv run pytest tests/test_telemetry.py -v
```

Expected: FAIL because `aimo_interp_infra.telemetry` does not exist.

- [ ] **Step 3: Implement telemetry with optional CUDA observation**

Create `src/aimo_interp_infra/telemetry.py`:

```python
from __future__ import annotations

import json
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator


@dataclass(frozen=True)
class TelemetryReceipt:
    schema: str
    label: str
    wall_seconds: float
    max_rss_kib: int
    cuda_peak_bytes: int | None


def _cuda_peak_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


@contextmanager
def measure_runtime(label: str) -> Iterator[SimpleNamespace]:
    holder = SimpleNamespace(receipt=None)
    start = time.perf_counter()
    try:
        yield holder
    finally:
        elapsed = time.perf_counter() - start
        usage = resource.getrusage(resource.RUSAGE_SELF)
        holder.receipt = TelemetryReceipt(
            schema="aimo-interp-telemetry/v0.1",
            label=label,
            wall_seconds=float(elapsed),
            max_rss_kib=int(usage.ru_maxrss),
            cuda_peak_bytes=_cuda_peak_bytes(),
        )


def write_receipt(receipt: TelemetryReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

Append to `runtime/README.md`:

````markdown
Telemetry is an execution receipt, not a model feature. Pre-gate scientific
code may not consume telemetry fields as robustness predictors.
````

- [ ] **Step 4: Run telemetry tests**

Run:

```bash
uv run pytest tests/test_telemetry.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit telemetry**

```bash
git add src/aimo_interp_infra/telemetry.py tests/test_telemetry.py runtime/README.md
git commit -m "feat: record non-scientific runtime telemetry"
```

---

### Task 7: Add immutable release registration and gate logic

**Files:**
- Create: `src/aimo_interp_infra/release_gate.py`
- Create: `scripts/register_release.py`
- Test: `tests/test_release_gate.py`
- Modify: `COMPETITION_STATE.md`
- Modify: `RESEARCH_LEDGER.md`

**Interfaces:**
- Consumes an already acquired official artifact or schema file plus its source URL.
- Produces:
  - `ArtifactRecord`
  - `record_artifact(path: Path, source: str, revision: str, acquired_at_utc: str) -> ArtifactRecord`
  - `register(registry_path: Path, kind: Literal["training_data", "cot_activation_interface"], record: ArtifactRecord) -> dict`
- Registration records bytes/provenance only. It does not inspect labels or select features.

- [ ] **Step 1: Write failing release-gate tests**

Create `tests/test_release_gate.py`:

```python
import hashlib
import json
from pathlib import Path

import pytest

from aimo_interp_infra.release_gate import (
    ArtifactAlreadyRegistered,
    record_artifact,
    register,
)


def _empty_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "aimo-interp-release-registry/v0.1",
                "training_data": None,
                "cot_activation_interface": None,
                "gate_open": False,
            }
        ),
        encoding="utf-8",
    )


def test_record_artifact_hashes_exact_bytes(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")

    record = record_artifact(
        artifact,
        source="https://example.test/artifact",
        revision="rev-a",
        acquired_at_utc="2026-09-04T12:00:00Z",
    )

    assert record.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert record.size_bytes == 3
    assert record.path_kind == "file"


def test_record_artifact_hashes_directory_tree_deterministically(tmp_path: Path):
    artifact = tmp_path / "activation-interface"
    artifact.mkdir()
    (artifact / "schema.json").write_text('{"shape":[1,2]}\n', encoding="utf-8")
    nested = artifact / "docs"
    nested.mkdir()
    (nested / "README.md").write_text("activation contract\n", encoding="utf-8")

    first = record_artifact(
        artifact,
        source="https://example.test/activation-interface",
        revision="commit-123",
        acquired_at_utc="2026-09-04T12:00:00Z",
    )
    second = record_artifact(
        artifact,
        source="https://example.test/activation-interface",
        revision="commit-123",
        acquired_at_utc="2026-09-04T12:00:00Z",
    )

    assert first.sha256 == second.sha256
    assert first.path_kind == "directory"
    assert first.size_bytes > 0


def test_gate_opens_only_after_both_artifacts_registered(tmp_path: Path):
    registry = tmp_path / "registry.json"
    _empty_registry(registry)

    training = tmp_path / "training.json"
    training.write_text("{}", encoding="utf-8")
    activation = tmp_path / "activation.json"
    activation.write_text("{}", encoding="utf-8")

    training_record = record_artifact(
        training,
        source="https://example.test/training",
        revision="train-rev",
        acquired_at_utc="2026-09-04T12:00:00Z",
    )
    activation_record = record_artifact(
        activation,
        source="https://example.test/activation",
        revision="activation-rev",
        acquired_at_utc="2026-09-04T12:01:00Z",
    )

    first = register(registry, "training_data", training_record)
    assert first["gate_open"] is False

    second = register(registry, "cot_activation_interface", activation_record)
    assert second["gate_open"] is True


def test_registration_is_immutable(tmp_path: Path):
    registry = tmp_path / "registry.json"
    _empty_registry(registry)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    record = record_artifact(
        artifact,
        source="https://example.test/training",
        revision="train-rev",
        acquired_at_utc="2026-09-04T12:00:00Z",
    )
    register(registry, "training_data", record)

    with pytest.raises(ArtifactAlreadyRegistered):
        register(registry, "training_data", record)
```

- [ ] **Step 2: Run tests and verify the release-gate module is missing**

Run:

```bash
uv run pytest tests/test_release_gate.py -v
```

Expected: FAIL because `aimo_interp_infra.release_gate` does not exist.

- [ ] **Step 3: Implement immutable artifact records and gate computation**

Create `src/aimo_interp_infra/release_gate.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


class ArtifactAlreadyRegistered(RuntimeError):
    """Raised when a frozen release slot would be overwritten."""


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    revision: str
    acquired_at_utc: str
    sha256: str
    size_bytes: int
    path_name: str
    path_kind: str


def _hash_path(path: Path) -> tuple[str, int, str]:
    digest = hashlib.sha256()
    if path.is_file():
        payload = path.read_bytes()
        digest.update(payload)
        return digest.hexdigest(), len(payload), "file"

    if not path.is_dir():
        raise FileNotFoundError(path)

    total_size = 0
    for file_path in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        payload = file_path.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(payload)
        total_size += len(payload)
    return digest.hexdigest(), total_size, "directory"


def record_artifact(
    path: Path,
    source: str,
    revision: str,
    acquired_at_utc: str,
) -> ArtifactRecord:
    sha256, size_bytes, path_kind = _hash_path(path)
    return ArtifactRecord(
        source=source,
        revision=revision,
        acquired_at_utc=acquired_at_utc,
        sha256=sha256,
        size_bytes=size_bytes,
        path_name=path.name,
        path_kind=path_kind,
    )


def register(
    registry_path: Path,
    kind: Literal["training_data", "cot_activation_interface"],
    record: ArtifactRecord,
) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry[kind] is not None:
        raise ArtifactAlreadyRegistered(f"{kind} already registered")

    registry[kind] = asdict(record)
    registry["gate_open"] = (
        registry["training_data"] is not None
        and registry["cot_activation_interface"] is not None
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry
```

- [ ] **Step 4: Add the explicit registration CLI**

Create `scripts/register_release.py`:

```python
import argparse
from datetime import datetime, timezone
from pathlib import Path

from aimo_interp_infra.release_gate import record_artifact, register

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=["training_data", "cot_activation_interface"],
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    acquired_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    acquired_at = acquired_at.replace("+00:00", "Z")
    record = record_artifact(
        args.artifact,
        source=args.source,
        revision=args.revision,
        acquired_at_utc=acquired_at,
    )
    registry = register(ROOT / "RELEASE_REGISTRY.json", args.kind, record)

    print(f"{args.kind}: sha256={record.sha256}")
    print(f"gate_open={registry['gate_open']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run release-gate tests**

Run:

```bash
uv run pytest tests/test_release_gate.py -v
```

Expected: PASS.

- [ ] **Step 6: Add explicit state text explaining what gate-open means**

Append to `COMPETITION_STATE.md`:

````markdown
## External gate semantics

`RELEASE_REGISTRY.json` may set `gate_open=true` only when both official gating
artifacts have been immutably registered.

`gate_open=true` authorizes only the **observational audit**.

It does not authorize:
- feature selection;
- hypothesis preregistration before the audit is complete;
- classifier training;
- competitive scientific submission;
- scientific execution.
````

Append to `RESEARCH_LEDGER.md`:

````markdown
## Release-registration rule

Release registration is custody, not scientific interpretation. Hashing and
recording an artifact does not establish its observational unit, leakage
boundaries, label semantics, or scientific jurisdiction.
````

- [ ] **Step 7: Commit release-gate custody**

```bash
git add src/aimo_interp_infra/release_gate.py scripts/register_release.py \
  tests/test_release_gate.py COMPETITION_STATE.md RESEARCH_LEDGER.md
git commit -m "feat: add immutable AIMO release gate custody"
```

---

### Task 8: Add a pre-gate verifier and CI that proves the project stops in the correct state

**Files:**
- Create: `scripts/verify_pre_gate.py`
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_pre_gate_state.py`
- Modify: `README.md`
- Modify: `COMPETITION_STATE.md`
- Modify: `RESEARCH_LEDGER.md`

**Interfaces:**
- Consumes all infrastructure state.
- Produces command `uv run scripts/verify_pre_gate.py` with exit code 0 only when:
  - upstream lock is syntactically valid;
  - release registry has not silently opened science;
  - forbidden scientific files are absent;
  - control solutions satisfy the callable contract;
  - unit tests pass separately in CI.

- [ ] **Step 1: Extend the state test with forbidden-file checks**

Append to `tests/test_pre_gate_state.py`:

```python
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
```

- [ ] **Step 2: Run the state tests**

Run:

```bash
uv run pytest tests/test_pre_gate_state.py -v
```

Expected: PASS before the verifier exists, because these tests inspect repository state only.

- [ ] **Step 3: Implement the pre-gate verifier**

Create `scripts/verify_pre_gate.py`:

```python
import importlib.util
import json
from pathlib import Path

from aimo_interp_infra.upstream import UpstreamLock

ROOT = Path(__file__).resolve().parents[1]


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    if lock.commit != "e98c489a98acb6c833588dca74228bee9782d5dd":
        raise SystemExit("unexpected upstream commit")

    registry = json.loads(
        (ROOT / "RELEASE_REGISTRY.json").read_text(encoding="utf-8")
    )
    if registry["gate_open"]:
        raise SystemExit(
            "external gate is open; pre-gate verifier must be replaced by "
            "the separately authorized observational-audit workflow"
        )

    forbidden = [
        ROOT / "solutions",
        ROOT / "features",
        ROOT / "models",
        ROOT / "PREREGISTRATION.md",
        ROOT / "SCIENTIFIC_RESULT.json",
    ]
    present = [path for path in forbidden if path.exists()]
    if present:
        raise SystemExit(f"scientific objects exist before gate: {present}")

    for control in ("all-true", "all-false"):
        path = ROOT / "controls" / control / "solution.py"
        module = _load_solution(path)
        function = getattr(module, "are_robust", None)
        if not callable(function):
            raise SystemExit(f"{path} has no callable are_robust")
        output = function("example/model", ["p1", "p2"])
        if len(output) != 2 or any(type(value) is not bool for value in output):
            raise SystemExit(f"{path} violates native-bool batch contract")

    print("PRE_GATE_INFRASTRUCTURE_VALID")
    print("SCIENTIFIC_EXECUTION_NOT_AUTHORIZED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the pre-gate verifier**

Run:

```bash
uv run scripts/verify_pre_gate.py
```

Expected:

```text
PRE_GATE_INFRASTRUCTURE_VALID
SCIENTIFIC_EXECUTION_NOT_AUTHORIZED
```

- [ ] **Step 5: Add CI**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: uv sync --frozen
      - run: uv run pytest -q
      - run: uv run scripts/verify_pre_gate.py
```

- [ ] **Step 6: Run the complete local verification sequence**

Run:

```bash
uv sync
uv run pytest -q
uv run scripts/verify_pre_gate.py
uv run scripts/verify_upstream.py
```

Expected:
- all tests PASS;
- `PRE_GATE_INFRASTRUCTURE_VALID`;
- `SCIENTIFIC_EXECUTION_NOT_AUTHORIZED`;
- frozen upstream commit verifies exactly.

- [ ] **Step 7: Update the public state to software-verified / baseline-pending**

Append to `COMPETITION_STATE.md` only after Step 6 passes:

````markdown
## Software verification state

```text
PRE-GATE SOFTWARE:           VERIFIED
OFFICIAL MODEL BASELINE:     PENDING ENVIRONMENT ACCEPTANCE
SCIENTIFIC FEATURE FAMILY:   NONE
ROBUSTNESS CLASSIFIER:       NONE
LEADERBOARD-DIRECTED TUNING: NONE
NEXT ACTION:                 OFFICIAL BASELINE ACCEPTANCE
```
````

Append to `RESEARCH_LEDGER.md` only after Step 6 passes:

````markdown
## Pre-gate infrastructure verification

**Observation:** Repository custody, official-harness bridging, deterministic
packaging, generic one-load batch lifecycle, telemetry, and release registration
passed the pre-gate verification suite.

**Status:** `PRE_GATE_SOFTWARE_VERIFIED / BASELINE_ACCEPTANCE_PENDING`

**Claim ceiling:** This is an engineering/custody result only. It provides no
evidence that any representation, uncertainty statistic, trajectory quantity,
or other feature predicts organizer-defined robustness.

**Next legal action:** Run the pinned official trained-probe baseline unchanged
in a compatible model-cache/GPU environment and record the acceptance receipt.
````

Append to `README.md`:

````markdown
## Current stop

After software verification succeeds but before the official model baseline is
accepted:

```text
PRE_GATE_SOFTWARE_VERIFIED -> BASELINE_ACCEPTANCE_PENDING
```

Do not create `solutions/`, a feature package, a classifier, or a scientific
preregistration. After official baseline acceptance, the repository may move to
`INFRASTRUCTURE_READY -> WAITING_FOR_EXTERNAL_GATE`.
````

- [ ] **Step 8: Commit the verified pre-gate software state**

```bash
git add scripts/verify_pre_gate.py .github/workflows/ci.yml \
  tests/test_pre_gate_state.py README.md COMPETITION_STATE.md RESEARCH_LEDGER.md
git commit -m "chore: verify AIMO pre-gate software state"
```

- [ ] **Step 9: Verify the software-state commit and CI state**

Run:

```bash
git status --short
git log --oneline --decorate -8
uv run pytest -q
uv run scripts/verify_pre_gate.py
```

Expected:
- `git status --short` prints nothing;
- tests pass;
- verifier prints the two expected lines;
- no scientific implementation exists.

Push the branch and require the GitHub Actions `ci` workflow to pass before advancing to Task 9.

---

### Task 9: Record official trained-probe baseline acceptance before declaring infrastructure ready

**Files:**
- Create: `src/aimo_interp_infra/baseline_receipt.py`
- Create: `scripts/reproduce_official_baseline.py`
- Create: `receipts/README.md`
- Test: `tests/test_baseline_receipt.py`
- Modify: `COMPETITION_STATE.md`
- Modify: `RESEARCH_LEDGER.md`
- Modify: `README.md`

**Interfaces:**
- Consumes:
  - verified pinned upstream checkout;
  - unmodified upstream solution `solutions/trained-probe`;
  - official public validation input/reference directories;
  - a compatible model-cache/GPU environment.
- Produces:
  - `BaselineReceipt`
  - JSON receipt containing exact upstream commit, solution path, score-contract fields, telemetry, and execution status.
- The receipt is an engineering acceptance artifact only. Baseline accuracy must not be consumed by scientific feature selection.

- [ ] **Step 1: Write failing receipt tests**

Create `tests/test_baseline_receipt.py`:

```python
import json
from pathlib import Path

from aimo_interp_infra.baseline_receipt import (
    BaselineReceipt,
    write_baseline_receipt,
)


def test_baseline_receipt_round_trips_exact_contract_fields(tmp_path: Path):
    receipt = BaselineReceipt(
        schema="aimo-interp-baseline-receipt/v0.1",
        upstream_commit="e98c489a98acb6c833588dca74228bee9782d5dd",
        solution_path="solutions/trained-probe",
        status="PASS",
        accuracy=0.5,
        coverage=1.0,
        invalid_predictions=0,
        wall_seconds=12.5,
        max_rss_kib=1024,
        cuda_peak_bytes=None,
    )
    path = tmp_path / "receipt.json"
    write_baseline_receipt(receipt, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["upstream_commit"] == (
        "e98c489a98acb6c833588dca74228bee9782d5dd"
    )
    assert payload["solution_path"] == "solutions/trained-probe"
    assert payload["coverage"] == 1.0
    assert payload["invalid_predictions"] == 0


def test_baseline_receipt_rejects_incomplete_predictions():
    try:
        BaselineReceipt(
            schema="aimo-interp-baseline-receipt/v0.1",
            upstream_commit="e98c489a98acb6c833588dca74228bee9782d5dd",
            solution_path="solutions/trained-probe",
            status="PASS",
            accuracy=0.5,
            coverage=0.9,
            invalid_predictions=0,
            wall_seconds=1.0,
            max_rss_kib=1,
            cuda_peak_bytes=None,
        )
    except ValueError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("PASS receipt must require complete coverage")
```

- [ ] **Step 2: Run the receipt tests and verify the module is missing**

Run:

```bash
uv run pytest tests/test_baseline_receipt.py -v
```

Expected: FAIL because `aimo_interp_infra.baseline_receipt` does not exist.

- [ ] **Step 3: Implement the acceptance receipt type**

Create `src/aimo_interp_infra/baseline_receipt.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineReceipt:
    schema: str
    upstream_commit: str
    solution_path: str
    status: str
    accuracy: float | None
    coverage: float | None
    invalid_predictions: int | None
    wall_seconds: float
    max_rss_kib: int
    cuda_peak_bytes: int | None

    def __post_init__(self) -> None:
        if self.status == "PASS":
            if self.coverage != 1.0:
                raise ValueError("PASS receipt requires coverage == 1.0")
            if self.invalid_predictions != 0:
                raise ValueError(
                    "PASS receipt requires invalid_predictions == 0"
                )


def write_baseline_receipt(receipt: BaselineReceipt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Add the unmodified-baseline reproduction script**

Create `scripts/reproduce_official_baseline.py`:

```python
import argparse
from pathlib import Path

from aimo_interp_infra.baseline_receipt import (
    BaselineReceipt,
    write_baseline_receipt,
)
from aimo_interp_infra.official_harness import run_official
from aimo_interp_infra.telemetry import measure_runtime
from aimo_interp_infra.upstream import UpstreamLock, verify_checkout

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".cache" / "getting-started"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=UPSTREAM / "data" / "val-sample" / "input",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=UPSTREAM / "data" / "val-sample" / "reference",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "receipts" / "official-trained-probe.json",
    )
    args = parser.parse_args()

    lock = UpstreamLock.from_json(ROOT / "UPSTREAM_LOCK.json")
    verify_checkout(lock, UPSTREAM)
    solution = UPSTREAM / "solutions" / "trained-probe"

    with measure_runtime("official-trained-probe") as telemetry:
        result = run_official(
            upstream_checkout=UPSTREAM,
            solution=solution,
            input_dir=args.input_dir,
            reference_dir=args.reference_dir,
            small=args.small,
        )

    runtime = telemetry.receipt
    receipt = BaselineReceipt(
        schema="aimo-interp-baseline-receipt/v0.1",
        upstream_commit=lock.commit,
        solution_path="solutions/trained-probe",
        status="PASS",
        accuracy=result.accuracy,
        coverage=result.coverage,
        invalid_predictions=result.invalid_predictions,
        wall_seconds=runtime.wall_seconds,
        max_rss_kib=runtime.max_rss_kib,
        cuda_peak_bytes=runtime.cuda_peak_bytes,
    )
    write_baseline_receipt(receipt, args.receipt)
    print(args.receipt)


if __name__ == "__main__":
    main()
```

Create `receipts/README.md`:

````markdown
# Acceptance Receipts

Receipts in this directory document infrastructure execution only.

The official trained-probe receipt records:
- exact upstream commit;
- exact unmodified upstream solution path;
- coverage;
- invalid-prediction count;
- accuracy as an uninterpreted scorer output;
- runtime and memory telemetry.

Accuracy in these receipts is not an optimization signal and must not be used to
choose a scientific feature family.
````

- [ ] **Step 5: Run receipt tests**

Run:

```bash
uv run pytest tests/test_baseline_receipt.py -v
```

Expected: PASS.

- [ ] **Step 6: Attempt the real official trained-probe acceptance run**

Run:

```bash
uv run scripts/materialize_upstream.py
cd .cache/getting-started
uv run scripts/import_hf_dataset.py
cd ../..
uv run scripts/reproduce_official_baseline.py
```

Two legal outcomes exist.

If the required cached model/GPU environment is available, expected:
- process exits 0;
- receipt status is `PASS`;
- `coverage == 1.0`;
- `invalid_predictions == 0`;
- receipt records exact pinned upstream commit.

If the model cache/GPU environment is not available, do **not** modify the
official baseline and do **not** substitute a different model. Record the state
as `ENVIRONMENT_BLOCKED` in `COMPETITION_STATE.md`; infrastructure remains
`PRE_GATE_SOFTWARE_VERIFIED`, not `INFRASTRUCTURE_READY`.

- [ ] **Step 7: On PASS only, freeze the infrastructure-ready state**

Replace the software-verification block in `COMPETITION_STATE.md` with:

````markdown
## Infrastructure completion state

```text
PRE-GATE INFRASTRUCTURE:     VERIFIED
OFFICIAL MODEL BASELINE:     PASS
SCIENTIFIC FEATURE FAMILY:   NONE
ROBUSTNESS CLASSIFIER:       NONE
LEADERBOARD-DIRECTED TUNING: NONE
PROJECT STATE:               WAITING_FOR_EXTERNAL_GATE
```
````

Append to `RESEARCH_LEDGER.md`:

````markdown
## Official baseline acceptance

**Observation:** The pinned official `solutions/trained-probe` baseline executed
without modification under a compatible environment with complete coverage and
zero invalid predictions.

**Status:** `INFRASTRUCTURE_READY / WAIT`

**Claim ceiling:** Baseline accuracy is not interpreted as evidence for or
against any scientific hypothesis in this repository.

**Next legal action:** Wait for both official gating artifacts. Once both are
registered, open only the observational-audit phase.
````

Replace the README stop block with:

````markdown
## Current stop

```text
INFRASTRUCTURE_READY -> WAITING_FOR_EXTERNAL_GATE
```

No scientific solution, feature family, classifier, or preregistration is
authorized until both official gating artifacts are registered and the
observational audit is completed.
````

- [ ] **Step 8: Commit the acceptance tooling and, if available, the PASS receipt**

If Step 6 produced a PASS receipt:

```bash
git add src/aimo_interp_infra/baseline_receipt.py \
  scripts/reproduce_official_baseline.py receipts/README.md \
  receipts/official-trained-probe.json tests/test_baseline_receipt.py \
  COMPETITION_STATE.md RESEARCH_LEDGER.md README.md
git commit -m "chore: accept pinned official AIMO baseline"
```

If Step 6 is environment-blocked, commit the tooling and blocked state without a
fabricated receipt:

```bash
git add src/aimo_interp_infra/baseline_receipt.py \
  scripts/reproduce_official_baseline.py receipts/README.md \
  tests/test_baseline_receipt.py COMPETITION_STATE.md
git commit -m "chore: record AIMO baseline environment block"
```

- [ ] **Step 9: Verify the final authority state**

For a PASS path:

```bash
uv run pytest -q
grep -F "WAITING_FOR_EXTERNAL_GATE" COMPETITION_STATE.md
test -f receipts/official-trained-probe.json
git status --short
```

Expected:
- all tests PASS;
- state is `WAITING_FOR_EXTERNAL_GATE`;
- receipt exists;
- working tree is clean.

For an environment-blocked path:

```bash
uv run pytest -q
grep -F "ENVIRONMENT_BLOCKED" COMPETITION_STATE.md
test ! -f receipts/official-trained-probe.json
git status --short
```

Expected:
- all tests PASS;
- no PASS receipt exists;
- infrastructure is not falsely declared complete;
- working tree is clean.

---

## Official Baseline Acceptance Note

Task 9 is the sole authority for official trained-probe acceptance. Do not run a
different baseline, modify the upstream baseline, substitute a proxy model, or
interpret baseline accuracy as scientific evidence.

A missing compatible model-cache/GPU environment yields
`ENVIRONMENT_BLOCKED`, not a guessed or simulated PASS.

---

## Implementation Completion Gate

The plan is complete only when all of the following hold simultaneously:

```text
UPSTREAM PIN                VERIFIED
OFFICIAL HARNESS BRIDGE     VERIFIED
OFFICIAL TRAINED-PROBE      PASS
ALL_TRUE CONTROL            VERIFIED
ALL_FALSE CONTROL           VERIFIED
DETERMINISTIC ZIP           VERIFIED BYTE-FOR-BYTE
ONE-LOAD BATCH LIFECYCLE    VERIFIED WITH TEST DOUBLE
TELEMETRY                    VERIFIED
RELEASE REGISTRY             CLOSED
SCIENTIFIC FEATURE FAMILY    NONE
ROBUSTNESS CLASSIFIER        NONE
SCIENTIFIC PREREGISTRATION  NOT AUTHORIZED
SCIENTIFIC EXECUTION        NOT AUTHORIZED
PROJECT STATE               WAITING_FOR_EXTERNAL_GATE
```

If Task 9 ends `ENVIRONMENT_BLOCKED`, this completion gate is not satisfied;
the repository remains `PRE_GATE_SOFTWARE_VERIFIED` and waits for a compatible
acceptance environment without opening science.

At that point, stop.

The next implementation object is **not** a model. It is a separately
authorized observational audit after both official gating artifacts are
registered.
