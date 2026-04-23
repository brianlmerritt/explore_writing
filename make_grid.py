"""
Read spec.yaml, expand the parameter grid, write data/grid.tsv.

Each row of grid.tsv represents one (prompt × backend × parameter-set) cell.
explore.py will draw `n_samples` generations per row.

Run this before explore.py. Inspect grid.tsv. Count rows. Estimate cost.
Only then proceed.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
SPEC = ROOT / "spec.yaml"
PROMPTS_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
GRID_PATH = DATA_DIR / "grid.tsv"

GRID_FIELDS = [
    "run_id",
    "prompt_id",
    "prompt_hash",
    "backend_label",
    "provider",
    "model",
    "temperature",
    "top_p",
    "top_k",
    "n_samples",
    "max_tokens",
    "system_hash",
]


def _read_prompt(prompt_id: str) -> str:
    path = PROMPTS_DIR / f"{prompt_id}.txt"
    if not path.exists():
        sys.exit(f"ERROR: prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _short_hash(s: str) -> str:
    """12-char hex hash, enough for distinguishing grid rows."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    if not SPEC.exists():
        sys.exit(f"ERROR: {SPEC} not found. Create it from the example in README.md.")

    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    prompts = spec["prompts"]
    backends = spec["backends"]
    params = spec["parameters"]
    n_samples = int(spec["n_samples"])
    max_tokens = int(spec["max_tokens"])
    system = spec.get("system") or ""
    system_hash = _short_hash(system) if system else ""

    # Pre-read and pre-hash all prompts (catches missing files before we start)
    prompt_hashes = {pid: _short_hash(_read_prompt(pid)) for pid in prompts}

    temperatures = params["temperature"]
    top_ps = params["top_p"]
    top_ks = params.get("top_k", [None])

    DATA_DIR.mkdir(exist_ok=True)

    rows = []
    run_id = 1
    for prompt_id, backend, T, top_p, top_k in itertools.product(
        prompts, backends, temperatures, top_ps, top_ks
    ):
        rows.append({
            "run_id": f"r{run_id:05d}",
            "prompt_id": prompt_id,
            "prompt_hash": prompt_hashes[prompt_id],
            "backend_label": backend["label"],
            "provider": backend["provider"],
            "model": backend["model"],
            "temperature": T,
            "top_p": top_p,
            "top_k": "" if top_k is None else top_k,
            "n_samples": n_samples,
            "max_tokens": max_tokens,
            "system_hash": system_hash,
        })
        run_id += 1

    with GRID_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRID_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    total_generations = len(rows) * n_samples
    print(f"Wrote {GRID_PATH} with {len(rows)} rows.")
    print(f"explore.py will produce {total_generations} generations "
          f"({n_samples} samples × {len(rows)} cells).")
    print()
    print("Inspect grid.tsv before running explore.py.")
    print("Sanity-check the total generation count against your API budget.")


if __name__ == "__main__":
    main()
