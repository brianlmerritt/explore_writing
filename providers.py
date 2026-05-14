"""
Thin adapter layer: normalizes generation calls across providers to a single
function signature.

    generate(provider, model, prompt, T, top_p, top_k, max_tokens, seed=None)
        -> GenerationResult

Four of the five supported backends speak OpenAI's wire format (OpenAI itself,
OpenRouter, Gemini's OpenAI-compatible endpoint, and any local LLM served via
Ollama/llama.cpp/vLLM/LM Studio). So there are really only two adapters:
OpenAI-shaped and Anthropic-shaped.

Parameters not supported by a provider raise UnsupportedParam rather than being
silently dropped — silent parameter dropping is how experiments become
unreproducible.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    ttft_ms: int
    tokens_per_sec: float
    reasoning_tokens: int
    raw_response_id: str
    # Stored for debugging / rerunning; not usually needed downstream.
    provider: str
    model: str


class UnsupportedParam(Exception):
    """Raised when a caller requests a parameter the provider can't honor."""


class ProviderConfigError(Exception):
    """Raised when a provider is requested but its API key / URL is missing."""


# ---------------------------------------------------------------------------
# Provider capability table
# ---------------------------------------------------------------------------
#
# "supports_top_k" is the one non-obvious capability — OpenAI's chat API
# doesn't accept top_k at all, while Anthropic and Gemini do.
#
# For "local" and "openrouter" we conservatively say top_k is unsupported
# via the OpenAI-compatible chat.completions endpoint. Some local servers
# (Ollama, llama.cpp) DO accept top_k via an "extra_body" escape hatch;
# if you need it, add a dedicated adapter rather than overloading this one.

CAPABILITIES = {
    "anthropic":  {"supports_top_k": True},
    "openai":     {"supports_top_k": False},
    "gemini":     {"supports_top_k": True},   # via google-genai-compat OpenAI endpoint
    "openrouter": {"supports_top_k": True},   # depends on underlying model
    "local":      {"supports_top_k": False},  # conservative default
}


# ---------------------------------------------------------------------------
# Client construction (lazy — only build clients for providers actually used)
# ---------------------------------------------------------------------------

_clients: dict = {}


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        val = int(raw)
    except ValueError:
        raise ProviderConfigError(f"{name} must be an integer, got: {raw!r}")
    if val <= 0:
        raise ProviderConfigError(f"{name} must be > 0, got: {val}")
    return val


def _thinking_budget_tokens(provider: str) -> Optional[int]:
    """Return provider-specific or global reasoning token budget if configured."""
    per_provider = {
        "openrouter": "OPENROUTER_THINKING_BUDGET_TOKENS",
        "local": "LOCAL_THINKING_BUDGET_TOKENS",
        "openai": "OPENAI_THINKING_BUDGET_TOKENS",
        "gemini": "GEMINI_THINKING_BUDGET_TOKENS",
    }
    specific = per_provider.get(provider)
    if specific:
        val = _env_int(specific)
        if val is not None:
            return val
    return _env_int("THINKING_BUDGET_TOKENS")


def _anthropic_client():
    if "anthropic" not in _clients:
        from anthropic import Anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderConfigError("ANTHROPIC_API_KEY not set in .env")
        _clients["anthropic"] = Anthropic(api_key=key)
    return _clients["anthropic"]


def _openai_compat_client(provider: str):
    """Build an OpenAI-SDK client pointed at the right base_url for the provider."""
    cache_key = f"openai_compat:{provider}"
    if cache_key in _clients:
        return _clients[cache_key]

    from openai import OpenAI

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderConfigError("OPENAI_API_KEY not set in .env")
        client = OpenAI(api_key=key)

    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderConfigError("GEMINI_API_KEY not set in .env")
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    elif provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ProviderConfigError("OPENROUTER_API_KEY not set in .env")
        client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )

    elif provider == "local":
        base_url = os.environ.get("LOCAL_BASE_URL")
        if not base_url:
            raise ProviderConfigError("LOCAL_BASE_URL not set in .env")
        # Local servers often want a dummy key; the SDK refuses an empty string.
        key = os.environ.get("LOCAL_API_KEY") or "local-dummy-key"
        client = OpenAI(api_key=key, base_url=base_url)

    else:
        raise ValueError(f"Unknown OpenAI-compatible provider: {provider}")

    _clients[cache_key] = client
    return client


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(
    provider: str,
    model: str,
    prompt: str,
    T: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: int,
    seed: Optional[int] = None,
    system: Optional[str] = None,
) -> GenerationResult:
    """
    Run one generation. Raises UnsupportedParam for parameters the provider
    can't honor (rather than dropping them silently).
    """
    provider = provider.lower()
    if provider not in CAPABILITIES:
        raise ValueError(f"Unknown provider: {provider!r}")

    caps = CAPABILITIES[provider]
    if top_k is not None and not caps["supports_top_k"]:
        raise UnsupportedParam(
            f"provider={provider!r} does not support top_k via this adapter; "
            f"row requested top_k={top_k}"
        )

    t0 = time.perf_counter()

    if provider == "anthropic":
        result = _generate_anthropic(model, prompt, T, top_p, top_k, max_tokens, system)
    else:
        # openai, gemini, openrouter, local — all share the OpenAI SDK shape
        result = _generate_openai_compat(
            provider, model, prompt, T, top_p, top_k, max_tokens, seed, system
        )

    result.latency_ms = int((time.perf_counter() - t0) * 1000)
    result.provider = provider
    result.model = model
    return result


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _generate_anthropic(
    model: str,
    prompt: str,
    T: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: int,
    system: Optional[str],
) -> GenerationResult:
    client = _anthropic_client()

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if T is not None:
        kwargs["temperature"] = T
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if system is not None:
        kwargs["system"] = system

    t0 = time.perf_counter()
    t_first = None
    text_parts = []
    
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            if t_first is None:
                t_first = time.perf_counter()
            text_parts.append(text)
            
        final_msg = stream.get_final_message()
        
    t_end = time.perf_counter()
    ttft_ms = int((t_first - t0) * 1000) if t_first else 0
    gen_time = t_end - (t_first if t_first else t0)
    
    output_tokens = final_msg.usage.output_tokens
    
    return GenerationResult(
        text="".join(text_parts),
        finish_reason=final_msg.stop_reason or "",
        input_tokens=final_msg.usage.input_tokens,
        output_tokens=output_tokens,
        latency_ms=0,  # filled in by caller
        ttft_ms=ttft_ms,
        tokens_per_sec=(output_tokens / gen_time) if gen_time > 0 and output_tokens > 0 else 0.0,
        reasoning_tokens=0,
        raw_response_id=final_msg.id,
        provider="",   # filled in by caller
        model="",      # filled in by caller
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, Gemini-compat, OpenRouter, local)
# ---------------------------------------------------------------------------

def _generate_openai_compat(
    provider: str,
    model: str,
    prompt: str,
    T: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: int,
    seed: Optional[int],
    system: Optional[str],
) -> GenerationResult:
    client = _openai_compat_client(provider)
    thinking_budget = _thinking_budget_tokens(provider)

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    # Most newer openai-compat providers support stream_options
    kwargs["stream_options"] = {"include_usage": True}
    
    if T is not None:
        kwargs["temperature"] = T
    if top_p is not None:
        kwargs["top_p"] = top_p
    if seed is not None:
        kwargs["seed"] = seed
    extra_body = {}
    if top_k is not None:
        extra_body["top_k"] = top_k

    # Provider-specific reasoning controls. Not all providers honor these.
    if thinking_budget is not None:
        if provider in {"openrouter", "local"}:
            extra_body["reasoning"] = {"max_tokens": thinking_budget}
        elif provider == "openai":
            # OpenAI exposes reasoning effort controls rather than strict token caps.
            kwargs["reasoning_effort"] = os.environ.get("OPENAI_REASONING_EFFORT", "low")
        elif provider == "gemini":
            extra_body["reasoning"] = {"max_tokens": thinking_budget}

    if extra_body:
        kwargs["extra_body"] = extra_body

    t0 = time.perf_counter()
    t_first = None
    text_parts = []
    reasoning_parts = []
    finish_reason = ""
    raw_response_id = ""
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Some OpenAI-compatible backends reject reasoning controls.
        if thinking_budget is not None and "extra_body" in kwargs:
            fallback_kwargs = dict(kwargs)
            fallback_extra = dict(fallback_kwargs.get("extra_body") or {})
            fallback_extra.pop("reasoning", None)
            if fallback_extra:
                fallback_kwargs["extra_body"] = fallback_extra
            else:
                fallback_kwargs.pop("extra_body", None)
            resp = client.chat.completions.create(**fallback_kwargs)
        else:
            raise
    for chunk in resp:
        if t_first is None and (chunk.choices or getattr(chunk, "usage", None)):
            t_first = time.perf_counter()
            
        raw_response_id = chunk.id or raw_response_id
        
        if chunk.choices:
            choice = chunk.choices[0]
            if getattr(choice.delta, "content", None):
                text_parts.append(choice.delta.content)
            
            # Catch "reasoning_content" (DeepSeek format) or "reasoning"
            reasoning = getattr(choice.delta, "reasoning_content", None) or getattr(choice.delta, "reasoning", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
                
        usage = getattr(chunk, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0)

    t_end = time.perf_counter()
    ttft_ms = int((t_first - t0) * 1000) if t_first else 0
    gen_time = t_end - (t_first if t_first else t0)
    
    text = "".join(text_parts)

    return GenerationResult(
        text=text,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=0,
        ttft_ms=ttft_ms,
        tokens_per_sec=(output_tokens / gen_time) if gen_time > 0 and output_tokens > 0 else 0.0,
        reasoning_tokens=reasoning_tokens,
        raw_response_id=raw_response_id,
        provider="",
        model="",
    )


# ---------------------------------------------------------------------------
# Sanity check script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: pyyaml is not installed. Please install it to run the sanity check.")

    spec_path = Path(__file__).parent / "spec.yaml"
    if not spec_path.exists():
        sys.exit("ERROR: spec.yaml not found.")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    backends = spec.get("backends", [])

    if not backends:
        sys.exit("No backends found in spec.yaml.")

    print(f"Checking {len(backends)} backends from spec.yaml...\n")

    for backend in backends:
        label = backend.get("label", "unknown")
        provider = backend.get("provider")
        model = backend.get("model")

        print(f"• [{label}] provider={provider!r}, model={model!r}")

        if provider not in CAPABILITIES:
            print(f"  ❌ ERROR: Unknown provider {provider!r}")
            continue

        try:
            if provider == "anthropic":
                client = _anthropic_client()
                available_models = [m.id for m in client.models.list().data]
            else:
                client = _openai_compat_client(provider)
                available_models = [m.id for m in client.models.list().data]

            if model in available_models:
                print("  ✅ OK (connected & model found)")
            else:
                print("  ⚠️  WARNING: Connected successfully, but model was not found in the provider's /models list.")
                
        except ProviderConfigError as e:
            print(f"  ❌ CONFIG ERROR: {e}")
        except Exception as e:
            print(f"  ❌ API/CONNECTION ERROR: {type(e).__name__}: {e}")
            
    print("\nCheck complete.")
