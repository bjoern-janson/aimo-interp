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

The local score bridge invokes the pinned upstream `scripts/run_local.py`
directly. It parses the official score object but does not reimplement ingestion
or scoring.

Allowed pre-gate controls:
- `controls/all-true`
- `controls/all-false`

`BatchModelExecutor` is a generic lifecycle primitive. It guarantees one loader
call and one releaser call per model batch. It intentionally has no
robustness-specific processor, feature extractor, threshold, or fallback.

`max_rss_bytes` is `null` when unsupported. CUDA telemetry is `null` without
CUDA; otherwise it records one reset-at-entry peak allocation per visible
device. Telemetry is never a scientific feature.

