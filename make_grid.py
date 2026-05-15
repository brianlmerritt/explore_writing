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
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
SPEC = ROOT / "spec.yaml"

_spec_raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
_remote = str(_spec_raw.get("REMOTE_FOLDER_PATH") or "").strip()
REMOTE = Path(_remote) if _remote else None

DATA_DIR             = (REMOTE / "data")            if REMOTE else ROOT / "data"
TEMP_PROMPTS_DIR     = (REMOTE / "temp_prompts")    if REMOTE else ROOT / "temp_prompts"
PROMPT_FOLDER        = (REMOTE / "prompts")         if REMOTE else ROOT / "prompt_examples"
PROMPT_RECIPE_FOLDER = (REMOTE / "prompt_recipes")  if REMOTE else ROOT / "prompt_recipe_examples"

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


def _expand_recipe(recipe_id: str) -> str:
    """Read a recipe yaml, concatenate its prompt files, return combined text."""
    recipe_path = PROMPT_RECIPE_FOLDER / f"{recipe_id}.yaml"
    if not recipe_path.exists():
        sys.exit(f"ERROR: recipe file not found: {recipe_path} - is it listed in spec.yaml but missing from {PROMPT_RECIPE_FOLDER}?")
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    parts = []
    for pid in recipe["prompts"]:
        p_path = PROMPT_FOLDER / f"{pid}.md"
        if not p_path.exists():
            sys.exit(f"ERROR: prompt file not found: {p_path}  (referenced by recipe {recipe_id})")
        parts.append(p_path.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def _short_hash(s: str) -> str:
    """12-char hex hash, enough for distinguishing grid rows."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    if not SPEC.exists():
        sys.exit(f"ERROR: {SPEC} not found. Create it from the example in README.md.")

    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    recipe_ids = spec["recipes"]
    backends = spec["backends"]
    n_samples = int(spec["n_samples"])
    max_tokens = int(spec["max_tokens"])
    system = spec.get("system") or ""
    system_hash = _short_hash(system) if system else ""

    # Expand each recipe into a combined prompt, write to temp_prompts/, pre-hash.
    # This catches missing files before we touch the grid.
    TEMP_PROMPTS_DIR.mkdir(exist_ok=True)
    prompt_hashes = {}
    for rid in recipe_ids:
        combined = _expand_recipe(rid)
        out_path = TEMP_PROMPTS_DIR / f"{rid}.md"
        out_path.write_text(combined, encoding="utf-8")
        prompt_hashes[rid] = _short_hash(combined)
    print(f"Wrote {len(recipe_ids)} combined prompt(s) to {TEMP_PROMPTS_DIR.relative_to(ROOT)}/")

    DATA_DIR.mkdir(exist_ok=True)

    rows = []
    run_id = 1
    for prompt_id in recipe_ids:
        for backend in backends:
            if not backend.get("use_in_grid", True):
                continue
                
            params = backend.get("parameters", {})
            temperatures = params.get("temperature", [None])
            top_ps = params.get("top_p", [None])
            top_ks = params.get("top_k", [None])

            for T, top_p, top_k in itertools.product(temperatures, top_ps, top_ks):
                rows.append({
                    "run_id": f"r{run_id:05d}",
                    "prompt_id": prompt_id,
                    "prompt_hash": prompt_hashes[prompt_id],
                    "backend_label": backend["label"],
                    "provider": backend["provider"],
                    "model": backend["model"],
                    "temperature": "" if T is None else T,
                    "top_p": "" if top_p is None else top_p,
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
