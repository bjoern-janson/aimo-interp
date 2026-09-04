from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


CHECKPOINT = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
KNOWN_ALIASES = {"qwen3-8b:low": CHECKPOINT}


@dataclass
class ModelRuntime:
    model: Any
    tokenizer: Any
    torch: Any
    input_device: Any


def resolve_checkpoint_model_id(model_id: str) -> str:
    return KNOWN_ALIASES.get(model_id, model_id)


@contextmanager
def isolated_torch_seed(torch_module: Any, seed: int) -> Iterator[None]:
    devices = (
        list(range(torch_module.cuda.device_count()))
        if torch_module.cuda.is_available()
        else []
    )
    with torch_module.random.fork_rng(devices=devices, enabled=True):
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
        yield


def load_model_runtime(model_id: str) -> ModelRuntime:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    checkpoint = resolve_checkpoint_model_id(model_id)
    loader_kwargs = {"local_files_only": True, "trust_remote_code": False}
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint, **loader_kwargs)
    except (OSError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **loader_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.float32,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.to("cpu")
    model.eval()
    embeddings = model.get_input_embeddings()
    input_device = (
        embeddings.weight.device
        if embeddings is not None
        else next(model.parameters()).device
    )
    return ModelRuntime(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        input_device=input_device,
    )


def generate_continuation(
    runtime: ModelRuntime, prompt_content: str, temperature: float, seed: int
) -> str:
    try:
        formatted = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("chat template failure") from exc
    if not isinstance(formatted, str):
        raise RuntimeError("chat template did not return text")
    inputs = runtime.tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    )
    inputs = {name: value.to(runtime.input_device) for name, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    model_eos_token_id = getattr(
        getattr(runtime.model, "generation_config", None), "eos_token_id", None
    )
    eos_token_id = (
        model_eos_token_id
        if isinstance(model_eos_token_id, int) and not isinstance(model_eos_token_id, bool)
        else runtime.tokenizer.eos_token_id
    )
    with runtime.torch.inference_mode(), isolated_torch_seed(runtime.torch, seed):
        sequences = runtime.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=0,
            num_beams=1,
            repetition_penalty=1.0,
            max_new_tokens=2048,
            use_cache=True,
            pad_token_id=runtime.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
    if (
        getattr(sequences, "ndim", None) != 2
        or sequences.shape[0] != 1
        or sequences.shape[1] < prompt_length
    ):
        raise RuntimeError("generation output shape violation")
    generated = sequences[0, prompt_length:].tolist()
    return runtime.tokenizer.decode(generated, skip_special_tokens=True)


def release_model_runtime(runtime: ModelRuntime) -> None:
    runtime.model = None
    runtime.tokenizer = None
    gc.collect()
    if runtime.torch.cuda.is_available():
        runtime.torch.cuda.empty_cache()
