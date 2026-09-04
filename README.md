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

## Materialize the frozen official starter

```bash
uv run scripts/materialize_upstream.py
uv run scripts/verify_upstream.py
```

The disposable checkout lives under `.cache/getting-started` and must remain
exactly at the commit recorded in `UPSTREAM_LOCK.json`.

## Deterministic packaging

```bash
uv run scripts/build_submission.py controls/all-false dist/all-false-small.zip --small
```

Rebuilding an unchanged source tree must reproduce the ZIP byte-for-byte.

## Current stop

After software verification succeeds but before the official model baseline is
accepted:

```text
PRE_GATE_SOFTWARE_VERIFIED -> BASELINE_ACCEPTANCE_PENDING
```

Do not create `solutions/`, a feature package, a classifier, or a scientific
preregistration. After official baseline acceptance, the repository may move to
`INFRASTRUCTURE_READY -> WAITING_FOR_EXTERNAL_GATE`.

