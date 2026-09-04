from aimo_interp_yolo.protocol import MemberProtocol
from aimo_interp_yolo.submission import predict_member


def member_a():
    return MemberProtocol(
        protocol_id="Y001-A", role="SELF_CONSISTENCY_PROBE",
        generation_count=3, generation_calls_per_problem=3,
        temperature_schedule=(0.7, 0.7, 0.7),
        top_p=0.95, top_k=0, max_new_tokens=2048,
        agreement_rule="UNANIMOUS", minimum_successful_extractions=3,
        seed_indices=(1, 2, 3), seed_namespace="YOLO001-B1",
    )


def control_false():
    return MemberProtocol(
        protocol_id="CTRL-F", role="LEADERBOARD_DIAGNOSTIC_CONTROL",
        generation_count=0, generation_calls_per_problem=0,
        temperature_schedule=(), top_p=None, top_k=None,
        max_new_tokens=None, agreement_rule="CONSTANT_FALSE",
        minimum_successful_extractions=0, seed_indices=(),
        seed_namespace="YOLO001-B1", constant_prediction=False,
    )


def test_model_loads_once_for_whole_batch():
    loads, releases, calls = [], [], []
    runtime = object()

    def loader(model_id):
        loads.append(model_id)
        return runtime

    def sampler(rt, prompt, temperature, seed):
        calls.append((rt, prompt, temperature, seed))
        return "FINAL: 4"

    def releaser(rt):
        releases.append(rt)

    result = predict_member(
        member_a(), "qwen3-8b:low", ["2+2?", "1+3?"],
        runtime_loader=loader, sample_generator=sampler, runtime_releaser=releaser,
    )

    assert result == [True, True]
    assert loads == ["qwen3-8b:low"]
    assert releases == [runtime]
    assert len(calls) == 6


def test_control_bypasses_runtime():
    def forbidden_loader(model_id):
        raise AssertionError("control must not load model")

    assert predict_member(
        control_false(), "qwen3-8b:low", ["a", "b"],
        runtime_loader=forbidden_loader,
    ) == [False, False]


def test_extraction_failure_is_false_not_infrastructure_failure():
    outputs = iter(["FINAL: 4", "no final line", "FINAL: 4"])

    result = predict_member(
        member_a(), "qwen3-8b:low", ["p"],
        runtime_loader=lambda _: object(),
        sample_generator=lambda *args: next(outputs),
        runtime_releaser=lambda _: None,
    )

    assert result == [False]


def test_infrastructure_failure_propagates_and_releases():
    released = []

    try:
        predict_member(
            member_a(), "qwen3-8b:low", ["p"],
            runtime_loader=lambda _: object(),
            sample_generator=lambda *args: (_ for _ in ()).throw(
                RuntimeError("synthetic generation failure")
            ),
            runtime_releaser=lambda rt: released.append(rt),
        )
    except RuntimeError as exc:
        assert "synthetic generation failure" in str(exc)
    else:
        raise AssertionError("infrastructure failure must propagate")

    assert len(released) == 1


def test_empty_batch_does_not_load_model():
    def forbidden_loader(model_id):
        raise AssertionError("empty batch must not load")

    assert predict_member(
        member_a(), "qwen3-8b:low", [], runtime_loader=forbidden_loader,
    ) == []
