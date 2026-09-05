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
activation interface have been released, registered with write-once,
content-addressed, Git-custodied provenance, and passed a separate
observational audit.

The repository contains pre-gate infrastructure plus a separately governed YOLO exploratory competition lane; scientific execution remains closed.

## Execution surfaces

LOCAL BASELINE REPRODUCTION:       ENVIRONMENT_BLOCKED
EXTERNAL CODABENCH CONTRACT SMOKE: PASSED
EXTERNAL SUBMISSION ID:            915072
EXTERNAL ACCURACY:                 0.47
EXTERNAL COVERAGE:                 1.0
EXTERNAL INVALID PREDICTIONS:      0

## YOLO exploratory competition

YOLO EXPLORATORY COMPETITION:      AUTHORIZED BY APPROVED DESIGN
YOLO001-B1 DESIGN:                 APPROVED
YOLO001-B1 PROTOCOL:               OPEN
YOLO001-B1 SUBMISSION:             NOT AUTHORIZED BEFORE CLOSURE
SCIENTIFIC AUTHORITY FROM YOLO:    NONE

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

## Gate-artifact custody

`RELEASE_REGISTRY.json` is the sole custody authority for training data and
the CoT activation interface. Registration is write-once through its API,
content-addressed, and Git-custodied; it is not history-rewrite-proof.
Registration is custody only. Before both records exist, inspect neither
training-data, label, grouping, nor activation contents.

## Terminal state

```text
INFRASTRUCTURE STATUS:                  ENVIRONMENT_BLOCKED
OFFICIAL_BASELINE_CONTRACT_SMOKE:       NOT ACCEPTED
BLOCKER:                                COMPATIBLE MODEL-CACHE/GPU ENVIRONMENT UNAVAILABLE
SCIENTIFIC FEATURE FAMILY:              NONE
ROBUSTNESS CLASSIFIER:                  NONE
LEADERBOARD-DIRECTED TUNING:            NONE
PROJECT STATE:                          ENVIRONMENT_BLOCKED
```

