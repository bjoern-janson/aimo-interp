from aimo_interp_yolo.core import (
    derive_seed, problem_with_suffix, extract_final_answer, normalize_answer, decide,
)
from aimo_interp_yolo.protocol import MemberProtocol


def member(rule: str, k: int, minimum: int) -> MemberProtocol:
    return MemberProtocol(
        protocol_id="TEST", role="SELF_CONSISTENCY_PROBE", generation_count=k,
        generation_calls_per_problem=k,
        temperature_schedule=tuple(0.7 for _ in range(k)), top_p=0.95, top_k=0,
        max_new_tokens=2048, agreement_rule=rule,
        minimum_successful_extractions=minimum,
        seed_indices=tuple(range(1, k + 1)), seed_namespace="YOLO001-B1",
    )


def test_seed_contract_exact_vectors():
    assert derive_seed("YOLO001-B1", "2+2?", 1) == 6872521650337045198
    assert derive_seed("YOLO001-B1", "2+2?", 2) == 744643422856651384
    assert derive_seed("YOLO001-B1", "Line 1\nLine 2", 1) == 7697382424971408085


def test_seed_rejects_zero_index():
    try:
        derive_seed("YOLO001-B1", "x", 0)
    except ValueError as exc:
        assert "one-based" in str(exc)
    else:
        raise AssertionError("zero sample index must fail")


def test_prompt_preserves_problem_verbatim():
    problem = "  x + y?\r\nkeep spacing  "
    content = problem_with_suffix(problem)
    assert content[len(problem):] == (
        "\n\nSolve the problem independently. Reason as needed.\n"
        "End your response with exactly one line:\nFINAL: <answer>"
    )
    assert content[:len(problem)] == problem


def test_extraction_uses_last_final_line_case_insensitively():
    assert extract_final_answer("FINAL: 2\nwork\n final: 4 ") == "4"
    assert extract_final_answer("FINAL:") is None
    assert extract_final_answer("answer: 4") is None


def test_normalization_is_syntactic_only():
    assert normalize_answer("  $\\sqrt{2}$  ") == "\\sqrt{2}"
    assert normalize_answer("  a   b. ") == "a b"
    assert normalize_answer("1/2") == "1/2"
    assert normalize_answer("0.5") == "0.5"
    assert normalize_answer("A") != normalize_answer("a")


def test_unanimity_and_supermajority_rules():
    assert decide(member("UNANIMOUS", 3, 3), ["4", "4", "4"]) is True
    assert decide(member("UNANIMOUS", 3, 3), ["4", "4", None]) is False
    assert decide(member("AT_LEAST_4_OF_5", 5, 4), ["4", "4", "4", "4", "5"]) is True
    assert decide(member("AT_LEAST_4_OF_5", 5, 4), ["4", "4", "4", None, "5"]) is False
