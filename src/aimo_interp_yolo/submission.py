from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .core import decide, derive_seed, extract_final_answer, problem_with_suffix
from .model_runtime import (
    generate_continuation,
    load_model_runtime,
    release_model_runtime,
)
from .protocol import MemberProtocol, load_member


def predict_member(
    member: MemberProtocol,
    model_id: str,
    problems: list[str],
    *,
    runtime_loader: Callable[[str], Any] | None = None,
    sample_generator: Callable[[Any, str, float, int], str] | None = None,
    runtime_releaser: Callable[[Any], None] | None = None,
) -> list[bool]:
    if not problems:
        return []
    if member.role == "LEADERBOARD_DIAGNOSTIC_CONTROL":
        if type(member.constant_prediction) is not bool:
            raise ValueError("control lacks native bool constant")
        return [member.constant_prediction for _ in problems]

    loader = runtime_loader or load_model_runtime
    sampler = sample_generator or generate_continuation
    releaser = runtime_releaser or release_model_runtime
    runtime = loader(model_id)
    try:
        output: list[bool] = []
        for problem in problems:
            prompt = problem_with_suffix(problem)
            extracted: list[str | None] = []
            for index, temperature in zip(
                member.seed_indices, member.temperature_schedule, strict=True
            ):
                seed = derive_seed(member.seed_namespace, problem, index)
                continuation = sampler(runtime, prompt, temperature, seed)
                extracted.append(extract_final_answer(continuation))
            output.append(bool(decide(member, extracted)))
        return output
    finally:
        releaser(runtime)


def are_robust_from_bundle(
    member_path: Path, model_id: str, problems: list[str]
) -> list[bool]:
    _, member = load_member(member_path)
    return predict_member(member, model_id, problems)
