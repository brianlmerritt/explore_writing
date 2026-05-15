"""
Read data/grid.tsv and prompts/, generate samples, append to data/generations.tsv.

Resumable: if generations.tsv already contains rows for (run_id, sample_idx),
those are skipped. Kill with Ctrl-C any time; rerun to pick up where you left off.

One row of generations.tsv per sample (so n_samples rows per grid row).
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path

import yaml

from providers import generate, UnsupportedParam, ProviderConfigError, _thinking_budget_tokens

ROOT = Path(__file__).parent
SPEC = ROOT / "spec.yaml"
PROMPTS_DIR = ROOT / "temp_prompts"
DATA_DIR = ROOT / "data"
GRID_PATH = DATA_DIR / "grid.tsv"
GEN_PATH = DATA_DIR / "generations.tsv"
_env_output = os.environ.get("OUTPUT_TEXT_FOLDER_PATH", "")
OUTPUT_TEXT_DIR = Path(_env_output) if _env_output and _env_output != "." else ROOT / "output_text"

GEN_FIELDS = [
    "run_id",
    "sample_idx",
    "timestamp",
    "provider",
    "model",
    "backend_label",
    "prompt_id",
    "prompt_hash",
    "temperature",
    "top_p",
    "top_k",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "ttft_ms",
    "tokens_per_sec",
    "reasoning_tokens",
    "raw_response_id",
    "status",           # ok | error | unsupported_param | config_error
    "error",            # error message if status != ok, else ""
    "output_text",      # filename (.md) containing the generation text
]


def _load_prompts(grid_rows: list[dict]) -> dict[str, str]:
    needed = {row["prompt_id"] for row in grid_rows}
    prompts = {}
    for pid in needed:
        path = PROMPTS_DIR / f"{pid}.md"
        if not path.exists():
            sys.exit(f"ERROR: prompt file missing: {path}")
        prompts[pid] = path.read_text(encoding="utf-8")
    return prompts


def _load_grid() -> list[dict]:
    if not GRID_PATH.exists():
        sys.exit(f"ERROR: {GRID_PATH} not found. Run make_grid.py first.")
    with GRID_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_existing_keys() -> set[tuple[str, int]]:
    """Return set of (run_id, sample_idx) already present in generations.tsv."""
    if not GEN_PATH.exists():
        return set()
    keys = set()
    with GEN_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            keys.add((row["run_id"], int(row["sample_idx"])))
    return keys


def _open_append_writer():
    """Open generations.tsv for append, writing header only if file is new."""
    new_file = not GEN_PATH.exists()
    f = GEN_PATH.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=GEN_FIELDS, delimiter="\t")
    if new_file:
        writer.writeheader()
        f.flush()
    return f, writer


def _strip_thinking(text: str) -> str:
    """Best-effort removal of chain-of-thought style scaffolding."""
    if not text:
        return text

    cleaned = text.strip()

    # Remove explicit reasoning tags when present.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # If the model provides an explicit final-answer marker, keep only the tail.
    marker = re.search(
        r"(?:^|\n)\s*(?:final answer|final output|answer|output|here(?:'s| is) the (?:story|output))\s*:\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    if marker:
        cleaned = cleaned[marker.end():].strip()

    lines = cleaned.splitlines()
    if not lines:
        return cleaned

    planning_re = re.compile(
        r"\b(we need to|let'?s|now we need|must|instruction|count words|draft|scene \d|check that)\b",
        re.IGNORECASE,
    )
    planning_hits = sum(1 for ln in lines if planning_re.search(ln))

    story_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if re.match(r"^Scene\s+\d+\s*[:\-]", s, flags=re.IGNORECASE):
            story_start = i
            break
        if re.match(r"^(Mother|Father|Cleo|Leo|[A-Z][a-z]{1,20})\s*:\s*", s):
            story_start = i
            break
        if s.startswith('"') and len(s) > 1:
            story_start = i
            break

    if planning_hits >= 3 and story_start is not None and story_start > 0:
        cleaned = "\n".join(lines[story_start:]).strip()

    return cleaned.strip()


def _write_output_file(run_id: str, sample_idx: int, text: str) -> str:
    OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{run_id}_{sample_idx}.md"
    path = OUTPUT_TEXT_DIR / name
    path.write_text(text, encoding="utf-8")
    return name


def _coerce(row: dict) -> dict:
    """Turn string values from grid.tsv into the right Python types."""
    return {
        **row,
        "temperature": None if row["temperature"] == "" else float(row["temperature"]),
        "top_p": None if row["top_p"] == "" else float(row["top_p"]),
        "top_k": None if row["top_k"] == "" else int(row["top_k"]),
        "n_samples": int(row["n_samples"]),
        "max_tokens": int(row["max_tokens"]),
    }


def main() -> None:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    system = spec.get("system") or None

    grid = [_coerce(r) for r in _load_grid()]
    prompts = _load_prompts(grid)
    done = _load_existing_keys()

    total_planned = sum(r["n_samples"] for r in grid)
    already_done = len(done)
    todo = total_planned - already_done
    print(f"Grid: {len(grid)} cells × n_samples → {total_planned} generations planned.")
    print(f"Already in generations.tsv: {already_done}. To generate now: {todo}.")
    if todo == 0:
        print("Nothing to do.")
        return

    f, writer = _open_append_writer()
    try:
        for row in grid:
            run_id = row["run_id"]
            prompt = prompts[row["prompt_id"]]
            n_samples = row["n_samples"]

            for sample_idx in range(n_samples):
                key = (run_id, sample_idx)
                if key in done:
                    continue

                # Base record shared between success and failure paths
                record = {
                    "run_id": run_id,
                    "sample_idx": sample_idx,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "provider": row["provider"],
                    "model": row["model"],
                    "backend_label": row["backend_label"],
                    "prompt_id": row["prompt_id"],
                    "prompt_hash": row["prompt_hash"],
                    "temperature": row["temperature"],
                    "top_p": row["top_p"],
                    "top_k": "" if row["top_k"] is None else row["top_k"],
                    "finish_reason": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "ttft_ms": 0,
                    "tokens_per_sec": 0.0,
                    "reasoning_tokens": 0,
                    "raw_response_id": "",
                    "status": "ok",
                    "error": "",
                    "output_text": "",
                }

                try:
                    result = generate(
                        provider=row["provider"],
                        model=row["model"],
                        prompt=prompt,
                        T=row["temperature"],
                        top_p=row["top_p"],
                        top_k=row["top_k"],
                        max_tokens=row["max_tokens"],
                        system=system,
                        thinking_budget=_thinking_budget_tokens(row["provider"], spec.get("thinking_budget_tokens")),
                        openai_reasoning_effort=spec.get("openai_reasoning_effort", "low"),
                    )
                    cleaned_text = _strip_thinking(result.text)
                    out_name = _write_output_file(run_id, sample_idx, cleaned_text)
                    record.update(
                        finish_reason=result.finish_reason,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        ttft_ms=result.ttft_ms,
                        tokens_per_sec=result.tokens_per_sec,
                        reasoning_tokens=result.reasoning_tokens,
                        raw_response_id=result.raw_response_id,
                        output_text=out_name,
                    )

                except UnsupportedParam as e:
                    record["status"] = "unsupported_param"
                    record["error"] = str(e)
                except ProviderConfigError as e:
                    record["status"] = "config_error"
                    record["error"] = str(e)
                except KeyboardInterrupt:
                    print("\n[!] Ctrl-C detected. Abandoning current request and exiting...")
                    sys.exit(130)  # Standard exit code for SIGINT
                except Exception as e:
                    # Catch-all: network errors, API errors, rate limits, etc.
                    # Logged so explore.py keeps moving through the grid.
                    record["status"] = "error"
                    record["error"] = f"{type(e).__name__}: {e}"

                writer.writerow(record)
                f.flush()  # persist immediately so Ctrl-C doesn't lose work

                marker = "✓" if record["status"] == "ok" else "✗"
                
                t_str = "null" if row["temperature"] is None else f"{row['temperature']:.2f}"
                p_str = "null" if row["top_p"] is None else f"{row['top_p']:.2f}"
                ttft_str = f"{record['ttft_ms']}ms"
                speed_str = f"{record['tokens_per_sec']:.1f}t/s"
                
                print(f"  {marker} {run_id}/{sample_idx}  {row['backend_label']:20s}  "
                      f"T={t_str} top_p={p_str}  "
                      f"time={ttft_str} speed={speed_str} "
                      f"[{record['status']}]")
    finally:
        f.close()

    print()
    print(f"Done. Results in {GEN_PATH}.")


if __name__ == "__main__":
    main()
