from contextlib import contextmanager

import pytest

from aimo_interp_yolo.model_runtime import (
    ModelRuntime,
    generate_continuation,
    isolated_torch_seed,
    resolve_checkpoint_model_id,
)


class FakeVector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))
        self.ndim = 2

    def to(self, device):
        return self

    def __getitem__(self, key):
        if key == 0:
            return FakeVector(self.values)
        if isinstance(key, tuple) and key[0] == 0:
            return FakeVector(self.values[key[1]])
        raise AssertionError(f"unexpected key: {key!r}")


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 99

    def __init__(self):
        self.chat_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        return "FORMATTED"

    def __call__(self, text, return_tensors, add_special_tokens):
        assert (text, return_tensors, add_special_tokens) == ("FORMATTED", "pt", False)
        return {"input_ids": FakeTensor([10, 11])}

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids == [20, 21]
        assert skip_special_tokens is True
        return "FINAL: 4"


class FakeModel:
    def __init__(self, generation_eos_token_id=None):
        self.generation_config = type(
            "GenerationConfig", (), {"eos_token_id": generation_eos_token_id}
        )()
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return FakeTensor([10, 11, 20, 21])


class FakeInferenceTorch:
    @contextmanager
    def inference_mode(self):
        yield


def test_alias_resolution_maps_only_the_known_alias():
    assert (
        resolve_checkpoint_model_id("qwen3-8b:low")
        == "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    )


def test_alias_resolution_passes_unknown_identifier_through_directly():
    assert resolve_checkpoint_model_id("other/model") == "other/model"


def test_isolated_seed_uses_all_visible_cuda_devices():
    calls = []

    class Cuda:
        def is_available(self):
            return True

        def device_count(self):
            return 2

        def manual_seed_all(self, seed):
            calls.append(("cuda_seed", seed))

    class Random:
        @contextmanager
        def fork_rng(self, devices, enabled):
            calls.append(("fork", devices, enabled))
            yield
            calls.append(("restore",))

    class Torch:
        cuda = Cuda()
        random = Random()

        def manual_seed(self, seed):
            calls.append(("manual_seed", seed))

    with isolated_torch_seed(Torch(), 17):
        calls.append(("body",))

    assert calls == [
        ("fork", [0, 1], True),
        ("manual_seed", 17),
        ("cuda_seed", 17),
        ("body",),
        ("restore",),
    ]


def test_generation_uses_exact_prompt_and_sampling_kwargs(monkeypatch):
    model, tokenizer = FakeModel(), FakeTokenizer()
    runtime = ModelRuntime(
        model=model,
        tokenizer=tokenizer,
        torch=FakeInferenceTorch(),
        input_device="cuda:0",
    )

    @contextmanager
    def fake_seed_scope(torch_module, seed):
        assert seed == 123
        yield

    monkeypatch.setattr(
        "aimo_interp_yolo.model_runtime.isolated_torch_seed", fake_seed_scope
    )

    text = generate_continuation(runtime, "PROBLEM+SUFFIX", 0.7, 123)

    assert text == "FINAL: 4"
    assert tokenizer.chat_calls == [
        ([{"role": "user", "content": "PROBLEM+SUFFIX"}], False, True)
    ]
    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.7
    assert model.kwargs["top_p"] == 0.95
    assert model.kwargs["top_k"] == 0
    assert model.kwargs["num_beams"] == 1
    assert model.kwargs["repetition_penalty"] == 1.0
    assert model.kwargs["max_new_tokens"] == 2048
    assert model.kwargs["use_cache"] is True


def test_generation_prefers_usable_model_generation_eos_token_id(monkeypatch):
    model = FakeModel(generation_eos_token_id=47)
    runtime = ModelRuntime(model, FakeTokenizer(), FakeInferenceTorch(), "cuda:0")

    @contextmanager
    def fake_seed_scope(torch_module, seed):
        yield

    monkeypatch.setattr(
        "aimo_interp_yolo.model_runtime.isolated_torch_seed", fake_seed_scope
    )

    generate_continuation(runtime, "P", 0.7, 1)

    assert model.kwargs["eos_token_id"] == 47


def test_generation_falls_back_to_tokenizer_eos_when_model_eos_is_unusable(monkeypatch):
    model = FakeModel(generation_eos_token_id=None)
    runtime = ModelRuntime(model, FakeTokenizer(), FakeInferenceTorch(), "cuda:0")

    @contextmanager
    def fake_seed_scope(torch_module, seed):
        yield

    monkeypatch.setattr(
        "aimo_interp_yolo.model_runtime.isolated_torch_seed", fake_seed_scope
    )

    generate_continuation(runtime, "P", 0.7, 1)

    assert model.kwargs["eos_token_id"] == 99


def test_generation_rejects_malformed_output(monkeypatch):
    class BadModel(FakeModel):
        def generate(self, **kwargs):
            return object()

    runtime = ModelRuntime(BadModel(), FakeTokenizer(), FakeInferenceTorch(), "cuda:0")

    @contextmanager
    def fake_seed_scope(torch_module, seed):
        yield

    monkeypatch.setattr(
        "aimo_interp_yolo.model_runtime.isolated_torch_seed", fake_seed_scope
    )

    with pytest.raises(RuntimeError, match="output shape"):
        generate_continuation(runtime, "P", 0.7, 1)
