from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

from .protocol import MemberProtocol

SUFFIX = (
    "\n\nSolve the problem independently. Reason as needed.\n"
    "End your response with exactly one line:\n"
    "FINAL: <answer>"
)


def derive_seed(namespace: str, problem: str, sample_index: int) -> int:
    if sample_index < 1:
        raise ValueError("sample_index must be one-based")
    payload = (
        namespace.encode("utf-8") + b"\x00" + problem.encode("utf-8")
        + b"\x00" + str(sample_index).encode("ascii")
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def problem_with_suffix(problem: str) -> str:
    return problem + SUFFIX


def extract_final_answer(continuation: str) -> str | None:
    for line in reversed(continuation.splitlines()):
        stripped = line.strip()
        if stripped.casefold().startswith("final:"):
            return normalize_answer(stripped.split(":", 1)[1])
    return None


def normalize_answer(answer: str) -> str | None:
    value = unicodedata.normalize("NFKC", answer).strip()
    if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    value = re.sub(r"\s+", " ", value).strip()
    if value.endswith("."):
        value = value[:-1].strip()
    return value or None


def decide(member: MemberProtocol, extracted: list[str | None]) -> bool:
    if len(extracted) != member.generation_count:
        raise ValueError("extraction count does not match generation_count")
    if member.agreement_rule == "UNANIMOUS":
        return bool(extracted) and all(x is not None for x in extracted) and len(set(extracted)) == 1
    if member.agreement_rule == "AT_LEAST_4_OF_5":
        counts = Counter(x for x in extracted if x is not None)
        return bool(counts) and max(counts.values()) >= 4
    raise ValueError(f"unsupported agreement rule: {member.agreement_rule}")
