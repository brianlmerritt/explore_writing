"""
Verbose one-off generation debugger for a single grid row.

Use this when you need to inspect streamed chunks, reasoning fields, usage,
and final response text for a specific generation.

Example:
    python explore_debug.py --run-id r00035 --sample-idx 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

from providers import CAPABILITIES, ProviderConfigError, UnsupportedParam, _anthropic_client, _openai_compat_client, _thinking_budget_tokens

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).parent
SPEC = ROOT / "spec.yaml"

_spec_raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
_remote = str(_spec_raw.get("REMOTE_FOLDER_PATH") or "").strip()
_REMOTE = Path(_remote) if _remote else None

DATA_DIR    = (_REMOTE / "data")         if _REMOTE else ROOT / "data"
PROMPTS_DIR = (_REMOTE / "temp_prompts") if _REMOTE else ROOT / "temp_prompts"

GRID_PATH = DATA_DIR / "grid.tsv"
GEN_PATH  = DATA_DIR / "generations.tsv"


def _coerce(row: dict) -> dict:
    return {
        **row,
        "temperature": None if row["temperature"] == "" else float(row["temperature"]),
        "top_p": None if row["top_p"] == "" else float(row["top_p"]),
        "top_k": None if row["top_k"] == "" else int(row["top_k"]),
        "n_samples": int(row["n_samples"]),
        "max_tokens": int(row["max_tokens"]),
    }


def _load_grid() -> list[dict]:
    if not GRID_PATH.exists():
        sys.exit(f"ERROR: {GRID_PATH} not found. Run make_grid.py first.")
    with GRID_PATH.open(encoding="utf-8", newline="") as f:
        return [_coerce(row) for row in csv.DictReader(f, delimiter="\t")]


def _load_prompt(prompt_id: str) -> str:
    path = PROMPTS_DIR / f"{prompt_id}.md"
    if not path.exists():
        sys.exit(f"ERROR: prompt file missing: {path}")
    return path.read_text(encoding="utf-8")


def _load_spec() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def _load_generation_row(run_id: str, sample_idx: int) -> dict | None:
    if not GEN_PATH.exists():
        return None
    with GEN_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["run_id"] == run_id and int(row["sample_idx"]) == sample_idx:
                return row
    return None


def _find_grid_row(run_id: str) -> dict:
    for row in _load_grid():
        if row["run_id"] == run_id:
            return row
    sys.exit(f"ERROR: run_id not found in grid.tsv: {run_id}")


def _print_row(row: dict) -> None:
    print("grid row:")
    for key in [
        "run_id",
        "prompt_id",
        "backend_label",
        "provider",
        "model",
        "temperature",
        "top_p",
        "top_k",
        "n_samples",
        "max_tokens",
        "prompt_hash",
        "system_hash",
    ]:
        print(f"  {key}: {row.get(key)!r}")


def _print_text_preview(label: str, text: str) -> None:
    print(label)
    print(text)


def _stream_anthropic(row: dict, prompt: str, system: str | None, thinking_budget: int | None) -> None:
    client = _anthropic_client()
    kwargs = {
        "model": row["model"],
        "max_tokens": row["max_tokens"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if row["temperature"] is not None:
        kwargs["temperature"] = row["temperature"]
    if row["top_p"] is not None:
        kwargs["top_p"] = row["top_p"]
    if row["top_k"] is not None:
        kwargs["top_k"] = row["top_k"]
    if system is not None:
        kwargs["system"] = system

    print("request:")
    print(f"  provider=anthropic model={row['model']!r}")
    print(f"  thinking_budget={thinking_budget!r}")
    print(f"  temperature={kwargs.get('temperature')!r} top_p={kwargs.get('top_p')!r} top_k={kwargs.get('top_k')!r}")
    print(f"  max_tokens={row['max_tokens']}")

    t0 = time.perf_counter()
    t_first = None
    text_parts: list[str] = []

    with client.messages.stream(**kwargs) as stream:
        for idx, text in enumerate(stream.text_stream, 1):
            if t_first is None:
                t_first = time.perf_counter()
            text_parts.append(text)
            print(f"[text {idx}] {text!r}")

        final_msg = stream.get_final_message()

    t_end = time.perf_counter()
    print("final:")
    print(f"  id={getattr(final_msg, 'id', '')!r}")
    print(f"  stop_reason={getattr(final_msg, 'stop_reason', '')!r}")
    usage = getattr(final_msg, 'usage', None)
    if usage is not None:
        print(f"  input_tokens={getattr(usage, 'input_tokens', 0)!r}")
        print(f"  output_tokens={getattr(usage, 'output_tokens', 0)!r}")
    blocks = getattr(final_msg, 'content', None) or []
    if blocks:
        print("  content blocks:")
        for i, block in enumerate(blocks, 1):
            print(f"    [{i}] type={getattr(block, 'type', type(block).__name__)!r} repr={block!r}")
    assembled = "".join(text_parts)
    print(f"  streamed_chars={len(assembled)}")
    print(f"  ttft_ms={int((t_first - t0) * 1000) if t_first else 0}")
    print(f"  total_ms={int((t_end - t0) * 1000)}")
    _print_text_preview("assembled text:", assembled)


def _stream_openai_compat(row: dict, prompt: str, system: str | None, thinking_budget: int | None, openai_reasoning_effort: str = "low") -> None:
    client = _openai_compat_client(row["provider"])

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": row["model"],
        "messages": messages,
        "max_tokens": row["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if row["temperature"] is not None:
        kwargs["temperature"] = row["temperature"]
    if row["top_p"] is not None:
        kwargs["top_p"] = row["top_p"]

    extra_body = {}
    if row["top_k"] is not None:
        extra_body["top_k"] = row["top_k"]

    if thinking_budget is not None:
        if row["provider"] in {"openrouter", "local", "gemini"}:
            extra_body["reasoning"] = {"max_tokens": thinking_budget}
        elif row["provider"] == "openai":
            kwargs["reasoning_effort"] = openai_reasoning_effort

    if extra_body:
        kwargs["extra_body"] = extra_body

    print("request:")
    print(f"  provider={row['provider']!r} model={row['model']!r}")
    print(f"  thinking_budget={thinking_budget!r}")
    print(f"  temperature={kwargs.get('temperature')!r} top_p={kwargs.get('top_p')!r}")
    print(f"  max_tokens={row['max_tokens']}")
    print(f"  extra_body={kwargs.get('extra_body')!r}")
    if "reasoning_effort" in kwargs:
        print(f"  reasoning_effort={kwargs['reasoning_effort']!r}")

    t0 = time.perf_counter()
    t_first = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = ""
    raw_response_id = ""
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        print(f"initial request failed: {type(exc).__name__}: {exc}")
        if thinking_budget is not None and "extra_body" in kwargs:
            fallback_kwargs = dict(kwargs)
            fallback_extra = dict(fallback_kwargs.get("extra_body") or {})
            fallback_extra.pop("reasoning", None)
            if fallback_extra:
                fallback_kwargs["extra_body"] = fallback_extra
            else:
                fallback_kwargs.pop("extra_body", None)
            print("retrying without reasoning control:")
            print(f"  extra_body={fallback_kwargs.get('extra_body')!r}")
            resp = client.chat.completions.create(**fallback_kwargs)
        else:
            raise

    for idx, chunk in enumerate(resp, 1):
        if t_first is None and (getattr(chunk, "choices", None) or getattr(chunk, "usage", None)):
            t_first = time.perf_counter()

        raw_response_id = getattr(chunk, "id", "") or raw_response_id
        choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
        content = getattr(choice.delta, "content", None) if choice else None
        reasoning = None
        if choice is not None:
            reasoning = getattr(choice.delta, "reasoning_content", None) or getattr(choice.delta, "reasoning", None)
        usage = getattr(chunk, "usage", None)

        print(f"[chunk {idx}] id={getattr(chunk, 'id', '')!r} finish_reason={getattr(choice, 'finish_reason', None)!r}")
        if content:
            print(f"  content={content!r}")
            text_parts.append(content)
        if reasoning:
            print(f"  reasoning={reasoning!r}")
            reasoning_parts.append(reasoning)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0)
            print(f"  usage prompt={input_tokens} completion={output_tokens} reasoning={reasoning_tokens}")
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason

    t_end = time.perf_counter()
    assembled = "".join(text_parts)
    print("final:")
    print(f"  raw_response_id={raw_response_id!r}")
    print(f"  finish_reason={finish_reason!r}")
    print(f"  input_tokens={input_tokens}")
    print(f"  output_tokens={output_tokens}")
    print(f"  reasoning_tokens={reasoning_tokens}")
    print(f"  streamed_chars={len(assembled)}")
    print(f"  reasoning_chars={sum(len(x) for x in reasoning_parts)}")
    print(f"  ttft_ms={int((t_first - t0) * 1000) if t_first else 0}")
    print(f"  total_ms={int((t_end - t0) * 1000)}")
    _print_text_preview("assembled text:", assembled)
    if reasoning_parts:
        _print_text_preview("assembled reasoning:", "".join(reasoning_parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug a single generation with detailed streamed response output.")
    parser.add_argument("--run-id", required=True, help="Grid run_id to reproduce.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index within the run_id.")
    args = parser.parse_args()

    row = _find_grid_row(args.run_id)
    generation = _load_generation_row(args.run_id, args.sample_idx)
    spec = _load_spec()
    system = spec.get("system") or None
    thinking_budget = _thinking_budget_tokens(row["provider"].lower(), spec.get("thinking_budget_tokens"))
    openai_reasoning_effort = spec.get("openai_reasoning_effort", "low")
    prompt = _load_prompt(row["prompt_id"])

    _print_row(row)
    if generation is not None:
        print("generation row:")
        for key in ["status", "error", "output_text", "finish_reason", "raw_response_id"]:
            print(f"  {key}: {generation.get(key)!r}")
    print(f"system_prompt_present={system is not None}")
    if system is not None:
        _print_text_preview("system prompt:", system)
    else:
        print("system prompt: <none>")
    _print_text_preview("prompt preview:", prompt)

    provider = row["provider"].lower()
    if provider not in CAPABILITIES:
        raise SystemExit(f"ERROR: unknown provider: {provider!r}")

    try:
        if provider == "anthropic":
            _stream_anthropic(row, prompt, system, thinking_budget)
        else:
            _stream_openai_compat(row, prompt, system, thinking_budget, openai_reasoning_effort)
    except UnsupportedParam as exc:
        raise SystemExit(f"ERROR: unsupported parameter: {exc}") from exc
    except ProviderConfigError as exc:
        raise SystemExit(f"ERROR: provider configuration: {exc}") from exc


if __name__ == "__main__":
    main()