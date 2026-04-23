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
    "openrouter": {"supports_top_k": False},  # conservative default
    "local":      {"supports_top_k": False},  # conservative default
}


# ---------------------------------------------------------------------------
# Client construction (lazy — only build clients for providers actually used)
# ---------------------------------------------------------------------------

_clients: dict = {}


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
    T: float,
    top_p: float,
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
            provider, model, prompt, T, top_p, max_tokens, seed, system
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
    T: float,
    top_p: float,
    top_k: Optional[int],
    max_tokens: int,
    system: Optional[str],
) -> GenerationResult:
    client = _anthropic_client()

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=T,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    if top_k is not None:
        kwargs["top_k"] = top_k
    if system is not None:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)

    # Assemble text from content blocks (ignore tool_use / thinking blocks if any)
    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    text = "".join(text_parts)

    return GenerationResult(
        text=text,
        finish_reason=resp.stop_reason or "",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=0,  # filled in by caller
        raw_response_id=resp.id,
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
    T: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int],
    system: Optional[str],
) -> GenerationResult:
    client = _openai_compat_client(provider)

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=T,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]

    usage = resp.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    return GenerationResult(
        text=choice.message.content or "",
        finish_reason=choice.finish_reason or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=0,
        raw_response_id=resp.id,
        provider="",
        model="",
    )
