# AIMO Interpretability Competition — Infrastructure-First Design

**Date:** 2026-09-04  
**Status:** DESIGN REVISION 1 / PRE-IMPLEMENTATION / NO SCIENTIFIC EXECUTION  
**Repository:** `bjoern-janson/aimo-interp`  
**Primary track:** Small Models Track  
**Secondary track:** Main Track  
**Pinned upstream starter:** `aimo-interp/getting-started@e98c489a98acb6c833588dca74228bee9782d5dd`

## 1. Purpose

Create a competition-specific, provenance-preserving execution lane for the AIMO Interpretability Challenge without spending scientific degrees of freedom before the two gating artifacts arrive:

1. official training data;
2. CoT activation interface.

The project exists to make the later scientific sequence fast, reproducible, and externally adjudicated. It does **not** import authority from Reach, Representation-Revision, Γ, L2-v1, or the representation-provenance concept.

The current transition is:

```text
ENTER
  -> SMALL
  -> wait for monitored release
  -> observe actual contract
  -> preregister
  -> execute
  -> external score
```

## 2. Scientific freeze

Before the training-data and activation-interface gate opens:

```text
competition decision       ENTER
track priority             SMALL
scientific target          ORDERED TRAJECTORY DIAGNOSTICS
label-replay strategy      CLOSED
feature ontology           NOT FROZEN
hypothesis                  NOT PREREGISTERED
experiment                  NOT OPENED
leaderboard optimization    NOT AUTHORIZED
```

No implementation before the gate may select, tune, or privilege trajectory features based on robustness labels.

The provisional future null hypothesis is recorded only as a target for later preregistration, not yet as an executable hypothesis:

> H0 candidate: ordered trajectory structure adds no predictive information beyond declared static/unordered controls.

The exact H0, controls, features, grouping scheme, estimand, and decision rule may be frozen only after the released observational universe has been audited.

## 3. External gate

The next scientific gate is:

```text
TRAINING DATA
      +
CoT ACTIVATION INTERFACE
```

After both are available, the legal sequence is:

```text
D_train
  -> Z_activation
  -> Y_label
  -> observational-unit audit
  -> leakage/grouping audit
  -> trajectory-dimension audit
  -> feature jurisdiction
  -> preregistration
  -> execution
```

The interface contract determines the observational universe. Theory does not get to silently expand it.

## 4. Scope before the gate

### Allowed

- Pin and record upstream versions.
- Reproduce the official local runtime contract.
- Build deterministic packaging and custody utilities.
- Verify submission-interface compatibility.
- Implement model-lifecycle plumbing that loads each model once per evaluation batch.
- Reproduce trivial controls and official baseline behavior strictly as execution checks.
- Measure runtime, memory, coverage, and failure behavior.
- Record upstream contract changes without interpreting them as scientific results.
- Register released gate artifacts by bytes, provenance, and revision only; registration is not permission to inspect their scientific contents.

### Forbidden before the gate

- Choose or tune trajectory features against robustness labels.
- Train a competition robustness classifier beyond reproducing supplied baselines.
- Use leaderboard feedback to choose feature families or hypotheses.
- Reconstruct hidden labels through runtime perturbation replay.
- Reopen Γ, L2-v1, Reach, or any other closed scientific lineage.
- Claim that ordered trajectory structure predicts robustness.
- Treat the public/warmup validation set as an optimization oracle.
- Inspect training-data, label, grouping, or activation contents before **both** gate artifacts are registered. Early registration remains custody only.

## 5. Repository architecture

The initial repository should stay deliberately small:

```text
README.md
UPSTREAM_LOCK.json
COMPETITION_STATE.md
RESEARCH_LEDGER.md

docs/
  superpowers/
    specs/
      2026-09-04-aimo-interp-design.md

vendor/
  README.md

runtime/
  README.md

tests/
  README.md
```

Only the design document is committed in the architectural stage. The other paths are implementation-plan targets, not files to create before plan approval.

### `UPSTREAM_LOCK.json`

Will record, at minimum:

- upstream repository;
- pinned starter commit;
- acquisition timestamp;
- upstream default branch;
- competition-site/contract references;
- later training-data revision;
- later activation-interface revision.

No upstream update is silently absorbed. Every update becomes an explicit provenance event.

### `COMPETITION_STATE.md`

Human-readable current contract and authority state. It distinguishes:

- historical proposal contract;
- current live execution contract;
- Discord/organizer clarifications;
- unresolved questions;
- scientifically authorized actions.

### `RESEARCH_LEDGER.md`

Append-only scientific decision log. At minimum:

```text
question
observation
status
claim ceiling
next legal action
```

Leaderboard scores belong here as adjudication events, not as tuning instructions.

### `RELEASE_REGISTRY.json`

Records gate artifacts as **write-once, content-addressed, Git-custodied** records. The API refuses replacement of an occupied slot; content hashes and preserved Git ancestry make later alteration detectable. This is not a claim of metaphysical or history-rewrite-proof immutability.

A training-data record or an activation-interface record may be registered before the other. Until both records exist, registration authorizes neither scientific content inspection nor any observational, label, grouping, feature, classifier, or trajectory work.

## 6. Execution architecture

The runtime layer should wrap—not reinterpret—the official Codabench contract.

Primary execution invariant:

```text
one are_robust(model_id, problems) call
  -> load matching model once
  -> process all supplied problems for that model
  -> return native bool for every problem
  -> release resources
```

The system must fail loudly in local validation rather than silently converting infrastructure failures into scientific predictions.

An unmodified official reference baseline is an `OFFICIAL_BASELINE_CONTRACT_SMOKE` only:

```text
OFFICIAL_BASELINE_CONTRACT_SMOKE
  != LIFECYCLE_CERTIFICATION
  != SCIENTIFIC_BASELINE
```

A PASS certifies only that the unmodified specimen is accepted by official ingestion, returns a complete valid output, and can execute in the measured environment. It does not certify target-model coverage, one-load lifecycle behavior, scientific appropriateness for Small, or a robustness claim.

Required execution properties:

- no inference-time network dependency;
- exact dependency compatibility with the official container;
- deterministic packaging;
- complete prediction coverage;
- explicit model-id routing;
- bounded GPU-memory lifecycle;
- no hidden fallback classifier unless prospectively specified and separately tested;
- cross-platform runtime and memory telemetry: RSS is explicitly unavailable (`None`) where unsupported, never fabricated as zero;
- per-device CUDA peak-allocation telemetry, with each measured device reset immediately before the block;
- runtime and peak-memory measurement.

## 7. Controls before scientific execution

Infrastructure validation may reproduce only non-scientific controls:

```text
ALL_TRUE
ALL_FALSE
OFFICIAL_REFERENCE_BASELINES
```

These establish plumbing, score direction, coverage, and environment compatibility. They do not authorize feature selection. The official reference baseline is recorded only as `OFFICIAL_BASELINE_CONTRACT_SMOKE` under the narrower semantics in Section 6.

A text-only or model-prior control may be added only after the live released training contract is audited, because the organizers have actively changed balancing and shortcut controls during the competition rollout.

## 8. Post-gate observational audit

When the gate opens, first produce a frozen observational audit containing:

### Training data

- exact revision/hash;
- row count;
- unique original problems;
- model/problem multiplicity;
- label counts overall and by model;
- continuous targets if present;
- duplicate/ancestry structure;
- mathematical family metadata if present;
- train/validation/test visibility boundaries.

### Activation interface

- exact revision/hash;
- models supported;
- layers exposed;
- token/time indices exposed;
- tensor shapes and dtypes;
- pooling/truncation rules;
- whether traces correspond to supplied/generated CoT;
- whether activation extraction changes generation;
- missing-data behavior;
- deterministic replay properties.

### Labels

- exact definition of robustness in the live dataset;
- aggregation across perturbations;
- threshold/majority/conjunctive rule;
- how repeated samples contribute;
- any outcome-conditioned admission/balancing process that affects prevalence interpretation.

No feature preregistration occurs until these are resolved sufficiently for a valid estimand.

## 9. Leakage and grouping discipline

Random row splitting is not assumed valid.

The post-gate audit must identify all grouping candidates, including:

- original problem;
- transformed descendants;
- problem family;
- model;
- dataset/source;
- symbolic-template ancestry if supplied;
- collection/admission batch.

At minimum, variants or repeated observations derived from one original problem may never cross a train/validation boundary.

The eventual validation scheme must distinguish leaderboard prediction from scientific generalization claims. A competition win does not by itself establish out-of-distribution robustness detection.

## 10. Future ablation jurisdiction

Only after preregistration may the experiment compare a ladder such as:

```text
B0  declared non-reasoning controls
B1  static endpoint representation
B2  unordered trajectory summaries
B3  ordered trajectory structure
```

The exact contents of B0–B3 remain intentionally unspecified before the activation interface is released.

The future falsification target is:

```text
B3 does not outperform B2 under the preregistered held-out design
  -> ordered trajectory structure adds no detected predictive value here.
```

If B3 does outperform B2, the maximum claim remains predictive:

> Ordered reasoning-state structure contains robustness-predictive information not captured by the declared static/unordered controls under this competition contract.

It does **not** establish causation, genuine reasoning, mechanism identity, representation-space expansion, or general deployment robustness.

## 11. Leaderboard discipline

The leaderboard is an external adjudicator, not a hyperparameter oracle.

Before the first competitive submission, freeze:

- feature family;
- preprocessing;
- model family;
- validation scheme;
- threshold/decision rule;
- fallback behavior;
- submission-count policy.

After score exposure, changes require an explicit new hypothesis or engineering correction. Leaderboard-directed feature shopping is prohibited.

Infrastructure smoke submissions using trivial controls or official baselines are exempt only insofar as they validate the execution contract and do not inform scientific feature selection.

## 12. Claim ceilings

Permanent distinctions:

```text
competition accuracy
  != mechanism identification
  != causal reasoning diagnosis
  != deployment calibration
  != representation provenance
  != representation-space expansion
```

```text
balanced curated evaluation
  != natural prevalence estimate
```

```text
activation predictiveness
  != activation causal necessity
```

```text
ordered trajectory signal
  != temporal causal direction
```

The live competition label is treated as organizer-defined finite counterfactual robustness, not as a universal definition of genuine reasoning.

## 13. Success criteria for the infrastructure phase

The pre-gate implementation is complete when all of the following are true:

1. upstream starter is pinned and provenance recorded;
2. local harness reproduces official ingestion/scoring on supplied sample data;
3. repository-owned batch lifecycle plumbing demonstrates one load and one release per `are_robust`-equivalent batch with a test double; this is not certified by the unmodified official baseline;
4. deterministic submission archives are reproducible byte-for-byte;
5. trivial controls produce complete valid Boolean outputs;
6. an unmodified official baseline, if environment-supported, passes the explicitly narrow `OFFICIAL_BASELINE_CONTRACT_SMOKE` acceptance;
7. cross-platform runtime, nullable RSS, and per-device CUDA telemetry are recorded with their measurement semantics;
8. no robustness feature family or classifier has been chosen from label evidence;
9. project state explicitly waits on training data + CoT activation interface.

## 14. Stop condition

After infrastructure completion, stop.

Do not open the scientific experiment until both gating artifacts are released and audited.

```text
infrastructure ready
  -> WAIT
  -> external release
  -> observational audit
  -> preregistration
```

The governing rule is:

> **Do not spend scientific degrees of freedom before the observational universe is known.**