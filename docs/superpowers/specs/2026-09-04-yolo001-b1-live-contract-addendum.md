# YOLO001-B1 Live-Contract and Plan-Consistency Addendum

**Status:** Binding implementation-contract addendum. It supplements, but does not replace, the approved YOLO001-B1 design at commit `2d2c362c3ea3f3970379d0520a9d2fc393df7127`.

## Scope

```text
LIVE-CONTRACT REPAIR
NO HEURISTIC CHANGE
NO LABEL INFORMATION
NO SCORE INFORMATION
NO SCIENTIFIC AUTHORITY
```

This addendum changes no B1 member, temperature, seed, prompt, extraction rule, decision rule, packaging envelope, or claim ceiling. `RELEASE_REGISTRY.json` remains the sole scientific-gate authority and remains closed.

## Organizer routing authority

The later competition-phase organizer announcement describes a final selection of five models, while naming only these four:

```text
openai/gpt-oss-120b
allenai/Olmo-3-7B-Think
deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
Qwen/Qwen3.5-4B
```

The unnamed fifth model is unknown. No checkpoint, alias, proxy, or fallback may be invented for it.

The entry point receives the organizer-supplied `model_id`. Runtime routing is therefore:

```python
checkpoint = KNOWN_ALIASES.get(model_id, model_id)
```

The only currently known alias is:

```text
qwen3-8b:low -> deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

The runtime attempts all other supplied identifiers directly through the standard offline Hugging Face loader with `local_files_only=True`. It must not maintain a DeepSeek-only whitelist. A cache/load failure remains an infrastructure failure and raises; it is never represented as a `False` prediction.

## Plan-consistency corrections

The executable plan must make all of the following exact:

1. The protocol has a first-class, exact `runtime_contract`.
2. Every member has `generation_calls_per_problem`, equal to `generation_count` for self-consistency members and zero for controls.
3. Protocol validation is constitutional: exact nested contract contents, no unspecified semantic keys, and recursive result-blindness over all dictionaries and lists.
4. Persistent custody contains logical repository source identity only; it contains no environment-specific absolute staging path.
5. `IMPLEMENTATION_COMMIT` is the already-verified current HEAD after the final implementation task. No empty "freeze" commit is created. The direct child closure commit contains the CLOSED protocol and custody objects.
6. Result append validates that `closure_commit_sha` names the actual closure commit: it exists, its parent equals custody `implementation_commit`, its tree has the exact CLOSED protocol and custody bytes/hashes, and its event ZIP digest matches custody.
7. Result schema is status-dependent. `SCORED` requires submission ID, submitted timestamp, observed timestamp, and all metrics. `INFRASTRUCTURE_FAILED` forbids metrics and may omit submission ID and timestamps. `WITHDRAWN_BEFORE_UPLOAD` requires null submission ID and submitted timestamp, requires an `observed_at_utc` withdrawal-record timestamp, and forbids metrics.

## EOS compatibility

Generation uses `model.generation_config.eos_token_id` when it is a usable integer; otherwise it uses `tokenizer.eos_token_id`. This is a loader compatibility detail, not a changed generation parameter.

## Unchanged prohibitions

The implementation must not generate organizer-like perturbations, reconstruct label rules, access labels or leaderboard data, use network access, or use external models. B1 remains original-problem-only target-model self-consistency.
