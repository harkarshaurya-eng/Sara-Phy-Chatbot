from __future__ import annotations

from pathlib import Path
from typing import Any

from src.formatting import messages_to_text, strip_thinking_tokens
from src.train_utils import (
    default_dtype,
    detect_runtime,
    get_system_prompt,
    model_architecture_hint,
    supports_4bit_quantization,
)

try:
    import torch
except Exception:  # pragma: no cover - optional at parse time
    torch = None


def load_tokenizer_for_model(model_name_or_path: str, trust_remote_code: bool = True):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"Could not find a tokenizer inside processor for {model_name_or_path}")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_quantization_config(runtime: dict[str, Any]):
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "QLoRA requested, but bitsandbytes support is unavailable. "
            "Install bitsandbytes on a CUDA machine or switch training_mode to 'lora'."
        ) from exc

    compute_dtype = default_dtype(runtime, prefer_bf16=True)
    if compute_dtype is None and torch is not None:
        compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    prefer_bf16: bool = True,
    load_in_4bit: bool = False,
    logger=None,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for loading the model.")

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    runtime = detect_runtime()
    architecture = model_architecture_hint(model_name_or_path)
    model_cls = AutoModelForImageTextToText if architecture == "image_text_to_text" else AutoModelForCausalLM
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": default_dtype(runtime, prefer_bf16=prefer_bf16),
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        if not supports_4bit_quantization(model_name_or_path, runtime):
            if logger:
                logger.warning(
                    "4-bit loading requested for %s, but runtime/model combination is unsupported. "
                    "Continuing without quantization.",
                    model_name_or_path,
                )
            load_in_4bit = False
        else:
            model_kwargs["quantization_config"] = _build_quantization_config(runtime)
            model_kwargs["device_map"] = "auto"
    elif runtime.get("accelerator") == "cuda":
        model_kwargs["device_map"] = "auto"

    if logger:
        logger.info("Loading base model %s with architecture=%s runtime=%s", model_name_or_path, architecture, runtime)

    try:
        model = model_cls.from_pretrained(model_name_or_path, **model_kwargs)
    except Exception as exc:
        if "gated" in str(exc).lower() or "access" in str(exc).lower():
            raise RuntimeError(
                f"Failed to load {model_name_or_path}. This is likely a gated model. "
                "Please accept the license at the model's HuggingFace page and ensure "
                "you are logged in with `huggingface_hub.login()`."
            ) from exc
        raise

    return model, runtime


def attach_adapter(base_model, adapter_path: str | Path, logger=None):
    from peft import PeftModel

    if logger:
        logger.info("Loading LoRA adapter from %s", adapter_path)
    return PeftModel.from_pretrained(base_model, str(adapter_path))


def load_chat_model(
    base_model_name: str,
    adapter_path: str | None = None,
    merged_model_path: str | None = None,
    trust_remote_code: bool = True,
    prefer_bf16: bool = True,
    load_in_4bit: bool = False,
    logger=None,
):
    load_path = merged_model_path or base_model_name
    tokenizer = load_tokenizer_for_model(load_path, trust_remote_code=trust_remote_code)
    model, runtime = load_base_model(
        load_path,
        trust_remote_code=trust_remote_code,
        prefer_bf16=prefer_bf16,
        load_in_4bit=load_in_4bit,
        logger=logger,
    )

    if adapter_path and not merged_model_path:
        model = attach_adapter(model, adapter_path, logger=logger)

    model.eval()
    return model, tokenizer, runtime


def _first_model_device(model):
    if hasattr(model, "device"):
        return model.device
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def ensure_system_message(messages: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system_prompt}] + list(messages)


def generate_chat_response(
    model,
    tokenizer,
    model_name: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.15,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for text generation.")

    system_prompt = system_prompt or get_system_prompt({"inference": {}})
    prepared_messages = ensure_system_message(messages, system_prompt=system_prompt)
    prompt_text = messages_to_text(
        prepared_messages,
        tokenizer=tokenizer,
        model_name=model_name,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    model_inputs = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True)
    device = _first_model_device(model)
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": max(temperature, 1e-5),
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }

    try:
        with torch.no_grad():
            outputs = model.generate(**model_inputs, **generate_kwargs)
    except RuntimeError as exc:
        error_msg = str(exc).lower()
        if "out of memory" in error_msg or "oom" in error_msg:
            # Try to free memory and give a helpful message
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"Out of memory during generation (max_new_tokens={max_new_tokens}). "
                "Try reducing max_new_tokens or using a smaller model."
            ) from exc
        raise

    prompt_tokens = int(model_inputs["input_ids"].shape[-1])
    generated_ids = outputs[0][prompt_tokens:]
    completion_tokens = int(generated_ids.shape[-1]) if len(generated_ids.shape) > 0 else 0
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = strip_thinking_tokens(text)

    return {
        "text": text.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "finish_reason": "stop",
    }


def interactive_chat(
    model,
    tokenizer,
    model_name: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    print("PhysicsGPT terminal chat. Type ':quit' to exit or ':reset' to clear history.")

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        if not user_message:
            continue
        if user_message.lower() in {":quit", ":exit"}:
            break
        if user_message.lower() == ":reset":
            history = [{"role": "system", "content": system_prompt}]
            print("History cleared.")
            continue

        history.append({"role": "user", "content": user_message})
        try:
            result = generate_chat_response(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                messages=history,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            )
            assistant_text = result["text"]
            history.append({"role": "assistant", "content": assistant_text})
            print(f"\nPhysicsGPT: {assistant_text}")
        except RuntimeError as exc:
            print(f"\n[Error] {exc}")
            # Remove the failed user message from history to keep it clean
            history.pop()
