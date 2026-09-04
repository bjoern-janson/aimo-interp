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

## Gate-artifact custody

`RELEASE_REGISTRY.json` is the sole custody authority for training data and
the CoT activation interface. Registration is write-once through its API,
content-addressed, and Git-custodied; it is not history-rewrite-proof.
Registration is custody only. Before both records exist, inspect neither
training-data, label, grouping, nor activation contents.

## Software verification state

```text
PRE-GATE SOFTWARE:           VERIFIED
OFFICIAL_BASELINE_CONTRACT_SMOKE:     PENDING ENVIRONMENT ACCEPTANCE
SCIENTIFIC FEATURE FAMILY:   NONE
ROBUSTNESS CLASSIFIER:       NONE
LEADERBOARD-DIRECTED TUNING: NONE
NEXT ACTION:                 OFFICIAL BASELINE CONTRACT-SMOKE ACCEPTANCE
```

