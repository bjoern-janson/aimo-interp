from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "aimo-interp-yolo-battery-protocol/v0.1"
BATTERY_ID = "YOLO001-B1"
RESULT_BEARING_KEYS = frozenset({
    "accuracy", "coverage", "invalid_predictions", "rank", "leaderboard_position",
    "submission_id", "codabench_submission_id", "score", "winner", "best", "selected",
})

RUNTIME_CONTRACT: dict[str, object] = {
    "entrypoint": "are_robust(model_id: str, problems: list[str]) -> list[bool]",
    "batch_invocation": "ONE_CALL_PER_MODEL_ID_BATCH",
    "return_contract": "NATIVE_BOOL_SAME_ORDER_ONE_PER_PROBLEM",
    "network": "UNAVAILABLE",
    "model_loader": "OFFLINE_HF_LOCAL_FILES_ONLY",
    "model_load_scope": "ONCE_PER_ARE_ROBUST_BATCH",
    "model_id_routing": {
        "default": "DIRECT_SUPPLIED_MODEL_ID",
        "known_aliases": {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"},
        "unknown_alias_policy": "NO_SPECULATIVE_ALIAS_DIRECT_LOAD",
        "load_failure_action": "RAISE",
    },
}
SHARED_GENERATION_CONTRACT: dict[str, object] = {
    "do_sample": True, "top_p": 0.95, "top_k": 0, "num_beams": 1,
    "repetition_penalty": 1.0, "max_new_tokens": 2048, "use_cache": True,
    "stop": "MODEL_TOKENIZER_EOS", "prediction_time_limit_seconds": 3600,
}
PROMPT_CONTRACT: dict[str, object] = {
    "problem_text": "EXACT_ORGANIZER_STRING_VERBATIM",
    "suffix": "\n\nSolve the problem independently. Reason as needed.\nEnd your response with exactly one line:\nFINAL: <answer>",
    "chat_template": "CHECKPOINT_NORMAL_CHAT_TEMPLATE", "add_generation_prompt": True,
    "extraction_source": "GENERATED_CONTINUATION_ONLY",
}
SEED_CONTRACT: dict[str, object] = {
    "algorithm": "SHA256_FIRST8_BIG_ENDIAN_MASK63",
    "payload_components": ["UTF8(seed_namespace)", "0x00", "UTF8(exact_problem_string)", "0x00", "ASCII(one_based_sample_index)"],
    "mask_hex": "0x7fffffffffffffff", "problem_normalization": "NONE",
}
EXTRACTION_CONTRACT: dict[str, object] = {
    "scan": "BOTTOM_TO_TOP_LAST_STRIPPED_CASE_INSENSITIVE_PREFIX_FINAL_COLON",
    "missing_or_empty": "EXTRACTION_FAILURE",
    "normalization": ["UNICODE_NFKC", "STRIP_OUTER_WHITESPACE", "REMOVE_ONE_WHOLE_ANSWER_DOLLAR_PAIR", "COLLAPSE_INTERNAL_WHITESPACE_TO_ASCII_SPACE", "REMOVE_ONE_TERMINAL_PERIOD"],
    "case": "PRESERVE", "mathematical_equivalence": "NONE",
}
FAILURE_CONTRACT: dict[str, object] = {
    "heuristic_observations": ["MISSING_FINAL_LINE", "EMPTY_EXTRACTED_ANSWER", "NORMALIZED_ANSWER_DISAGREEMENT"],
    "infrastructure_failures": ["MODEL_LOAD_FAILURE", "TOKENIZER_LOAD_FAILURE", "UNSUPPORTED_MODEL_ALIAS", "CUDA_OOM", "GENERATION_EXCEPTION", "INVALID_MODEL_RETURN", "OUTPUT_SHAPE_VIOLATION"],
    "infrastructure_failure_action": "RAISE",
}
RESERVE_POLICY: dict[str, object] = {
    "daily_budget": 10, "frozen_member_slots": 8, "reserve_slots": 2,
    "identical_byte_retry": True, "changed_bytes_same_member": False,
    "distinct_reserve_member_requires_prefreeze": True,
}
CLAIM_CEILING: dict[str, object] = {
    "scientific_preregistration": False, "scientific_evidence_status": "NONE",
    "public_development_label_use": "NONE", "training_label_use": "NONE",
    "leaderboard_tuning_within_battery": "NONE", "mechanism_claim": "NONE",
}
FROZEN_ROOT_SCALARS = {
    "scientific_authority": "NONE",
    "leaderboard_tuning_within_battery": "NONE",
    "public_development_label_use": "NONE",
    "training_label_use": "NONE",
    "seed_namespace": BATTERY_ID,
}


@dataclass(frozen=True)
class MemberProtocol:
    protocol_id: str
    role: str
    generation_count: int
    generation_calls_per_problem: int
    temperature_schedule: tuple[float, ...]
    top_p: float | None
    top_k: int | None
    max_new_tokens: int | None
    agreement_rule: str
    minimum_successful_extractions: int
    seed_indices: tuple[int, ...]
    seed_namespace: str
    constant_prediction: bool | None = None


@dataclass(frozen=True)
class BatteryProtocol:
    schema: str
    battery_id: str
    lane: str
    track: str
    protocol_state: str
    scientific_authority: str
    leaderboard_tuning_within_battery: str
    public_development_label_use: str
    training_label_use: str
    seed_namespace: str
    runtime_contract: dict[str, object]
    shared_generation_contract: dict[str, object]
    prompt_contract: dict[str, object]
    seed_contract: dict[str, object]
    extraction_contract: dict[str, object]
    failure_contract: dict[str, object]
    members: tuple[MemberProtocol, ...]
    reserve_policy: dict[str, object]
    claim_ceiling: dict[str, object]


FROZEN = {
    "Y001-A": ("SELF_CONSISTENCY_PROBE", 3, 3, (0.7, 0.7, 0.7), "UNANIMOUS", 3, None),
    "Y001-B": ("SELF_CONSISTENCY_PROBE", 5, 5, (0.7, 0.7, 0.7, 0.7, 0.7), "UNANIMOUS", 5, None),
    "Y001-C": ("SELF_CONSISTENCY_PROBE", 5, 5, (0.7, 0.7, 0.7, 0.7, 0.7), "AT_LEAST_4_OF_5", 4, None),
    "Y001-D": ("SELF_CONSISTENCY_PROBE", 3, 3, (0.4, 0.4, 0.4), "UNANIMOUS", 3, None),
    "Y001-E": ("SELF_CONSISTENCY_PROBE", 3, 3, (1.0, 1.0, 1.0), "UNANIMOUS", 3, None),
    "Y001-F": ("SELF_CONSISTENCY_PROBE", 4, 4, (0.4, 0.4, 0.9, 0.9), "UNANIMOUS", 4, None),
    "CTRL-T": ("LEADERBOARD_DIAGNOSTIC_CONTROL", 0, 0, (), "CONSTANT_TRUE", 0, True),
    "CTRL-F": ("LEADERBOARD_DIAGNOSTIC_CONTROL", 0, 0, (), "CONSTANT_FALSE", 0, False),
}
EXPECTED_IDS = tuple(FROZEN)
MEMBER_KEYS = frozenset(MemberProtocol.__dataclass_fields__)
ROOT_KEYS = frozenset(BatteryProtocol.__dataclass_fields__)


def _check_result_blind(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in RESULT_BEARING_KEYS:
                raise ValueError(f"result-bearing key is forbidden: {key}")
            _check_result_blind(nested)
    elif isinstance(value, list):
        for nested in value:
            _check_result_blind(nested)


def _expect_keys(payload: dict[str, object], expected: frozenset[str], context: str) -> None:
    missing = expected - payload.keys()
    extra = payload.keys() - expected
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} unexpected keys: {sorted(extra)}")


def _same_shape_and_values(value: object, expected: object, context: str) -> None:
    if type(value) is not type(expected):
        raise ValueError(f"{context} has invalid type")
    if isinstance(expected, dict):
        assert isinstance(value, dict)
        _expect_keys(value, frozenset(expected), context)
        for key, expected_value in expected.items():
            _same_shape_and_values(value[key], expected_value, f"{context}.{key}")
    elif isinstance(expected, list):
        assert isinstance(value, list)
        if len(value) != len(expected):
            raise ValueError(f"{context} has invalid length")
        for index, (actual, expected_item) in enumerate(zip(value, expected, strict=True)):
            _same_shape_and_values(actual, expected_item, f"{context}[{index}]")
    elif value != expected:
        raise ValueError(f"{context} does not match frozen contract")


def _native(value: object, expected: type[Any], field: str, *, nullable: bool = False) -> Any:
    if nullable and value is None:
        return None
    if type(value) is not expected:
        raise ValueError(f"{field} must be a native {expected.__name__}")
    return value


def member_from_dict(payload: dict[str, object]) -> MemberProtocol:
    _expect_keys(payload, MEMBER_KEYS, "member")
    schedule = _native(payload["temperature_schedule"], list, "temperature_schedule")
    seeds = _native(payload["seed_indices"], list, "seed_indices")
    if any(type(value) not in (int, float) or isinstance(value, bool) for value in schedule):
        raise ValueError("temperature_schedule must contain native numbers")
    if any(type(value) is not int for value in seeds):
        raise ValueError("seed_indices must contain native ints")
    return MemberProtocol(
        protocol_id=_native(payload["protocol_id"], str, "protocol_id"),
        role=_native(payload["role"], str, "role"),
        generation_count=_native(payload["generation_count"], int, "generation_count"),
        generation_calls_per_problem=_native(payload["generation_calls_per_problem"], int, "generation_calls_per_problem"),
        temperature_schedule=tuple(float(value) for value in schedule),
        top_p=_native(payload["top_p"], float, "top_p", nullable=True),
        top_k=_native(payload["top_k"], int, "top_k", nullable=True),
        max_new_tokens=_native(payload["max_new_tokens"], int, "max_new_tokens", nullable=True),
        agreement_rule=_native(payload["agreement_rule"], str, "agreement_rule"),
        minimum_successful_extractions=_native(payload["minimum_successful_extractions"], int, "minimum_successful_extractions"),
        seed_indices=tuple(seeds),
        seed_namespace=_native(payload["seed_namespace"], str, "seed_namespace"),
        constant_prediction=_native(payload["constant_prediction"], bool, "constant_prediction", nullable=True),
    )


def validate_protocol(protocol: BatteryProtocol) -> None:
    if (protocol.schema, protocol.battery_id, protocol.lane, protocol.track) != (
        SCHEMA, BATTERY_ID, "YOLO_EXPLORATORY_COMPETITION", "small",
    ):
        raise ValueError("invalid frozen battery identity")
    if protocol.protocol_state not in {"OPEN", "CLOSED"}:
        raise ValueError("invalid protocol state")
    if any(getattr(protocol, field) != expected for field, expected in FROZEN_ROOT_SCALARS.items()):
        raise ValueError("frozen root contract mismatch")
    if [member.protocol_id for member in protocol.members] != list(EXPECTED_IDS):
        if len({member.protocol_id for member in protocol.members}) != len(protocol.members):
            raise ValueError("duplicate protocol_id")
        raise ValueError("invalid ordered protocol IDs")
    for member in protocol.members:
        actual = (member.role, member.generation_count, member.generation_calls_per_problem,
                  member.temperature_schedule, member.agreement_rule,
                  member.minimum_successful_extractions, member.constant_prediction)
        if FROZEN[member.protocol_id] != actual:
            raise ValueError("frozen member mismatch")
        if member.role == "SELF_CONSISTENCY_PROBE":
            if (member.top_p, member.top_k, member.max_new_tokens, member.seed_indices,
                    member.seed_namespace) != (0.95, 0, 2048,
                                               tuple(range(1, member.generation_count + 1)), BATTERY_ID):
                raise ValueError("invalid self-consistency member contract")
        elif (member.top_p, member.top_k, member.max_new_tokens, member.seed_indices,
              member.seed_namespace) != (None, None, None, (), BATTERY_ID):
            raise ValueError("invalid control member contract")


def load_protocol(path: Path) -> BatteryProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol must be an object")
    _check_result_blind(payload)
    _expect_keys(payload, ROOT_KEYS, "protocol")
    for field, expected in {
        "runtime_contract": RUNTIME_CONTRACT,
        "shared_generation_contract": SHARED_GENERATION_CONTRACT,
        "prompt_contract": PROMPT_CONTRACT,
        "seed_contract": SEED_CONTRACT,
        "extraction_contract": EXTRACTION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "reserve_policy": RESERVE_POLICY,
        "claim_ceiling": CLAIM_CEILING,
    }.items():
        _same_shape_and_values(payload[field], expected, field)
    members_payload = payload["members"]
    if not isinstance(members_payload, list):
        raise ValueError("members must be a list")
    members = tuple(member_from_dict(member) if isinstance(member, dict) else (_ for _ in ()).throw(ValueError("member must be an object")) for member in members_payload)
    protocol = BatteryProtocol(
        schema=_native(payload["schema"], str, "schema"), battery_id=_native(payload["battery_id"], str, "battery_id"),
        lane=_native(payload["lane"], str, "lane"), track=_native(payload["track"], str, "track"),
        protocol_state=_native(payload["protocol_state"], str, "protocol_state"),
        scientific_authority=_native(payload["scientific_authority"], str, "scientific_authority"),
        leaderboard_tuning_within_battery=_native(payload["leaderboard_tuning_within_battery"], str, "leaderboard_tuning_within_battery"),
        public_development_label_use=_native(payload["public_development_label_use"], str, "public_development_label_use"),
        training_label_use=_native(payload["training_label_use"], str, "training_label_use"),
        seed_namespace=_native(payload["seed_namespace"], str, "seed_namespace"),
        runtime_contract=payload["runtime_contract"], shared_generation_contract=payload["shared_generation_contract"],
        prompt_contract=payload["prompt_contract"], seed_contract=payload["seed_contract"],
        extraction_contract=payload["extraction_contract"], failure_contract=payload["failure_contract"],
        members=members, reserve_policy=payload["reserve_policy"], claim_ceiling=payload["claim_ceiling"],
    )
    validate_protocol(protocol)
    return protocol


def load_member(path: Path) -> tuple[str, MemberProtocol]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "aimo-interp-yolo-member/v0.1":
        raise ValueError("invalid member schema")
    if payload.get("battery_id") != BATTERY_ID:
        raise ValueError("invalid member battery")
    member_payload = payload.get("member")
    if not isinstance(member_payload, dict):
        raise ValueError("invalid member payload")
    member = member_from_dict(member_payload)
    actual = (member.role, member.generation_count, member.generation_calls_per_problem,
              member.temperature_schedule, member.agreement_rule,
              member.minimum_successful_extractions, member.constant_prediction)
    if FROZEN.get(member.protocol_id) != actual:
        raise ValueError("frozen member mismatch")
    return BATTERY_ID, member


def member_by_id(protocol: BatteryProtocol, protocol_id: str) -> MemberProtocol:
    for member in protocol.members:
        if member.protocol_id == protocol_id:
            return member
    raise KeyError(protocol_id)
