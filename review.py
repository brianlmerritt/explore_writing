"""
Read data/generations.tsv, score each generation against rubric.yaml,
append to data/reviews.tsv.

Separate from explore.py so you can:
  - re-score with a different rubric without regenerating outputs
  - score with a different reviewer model
  - iterate on review criteria cheaply

Resumable the same way explore.py is: rows already present are skipped.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import yaml

from providers import generate, UnsupportedParam, ProviderConfigError

ROOT = Path(__file__).parent
RUBRIC = ROOT / "rubric.yaml"
DATA_DIR = ROOT / "data"
GEN_PATH = DATA_DIR / "generations_pass1.tsv"
REV_PATH = DATA_DIR / "reviews.tsv"

REV_FIELDS = [
    "run_id",
    "sample_idx",
    "timestamp",
    "reviewer_provider",
    "reviewer_model",
    "rubric_hash",       # so you can tell which rubric version produced these scores
    "status",            # ok | parse_error | error
    "error",
    "scores_json",       # {"concreteness": 4, "freshness": 3, ...}
    "notes",             # short free-text justification from the reviewer
]

REVIEW_SYSTEM = """You are a critical reader scoring a piece of writing against a rubric.
You are not the writer's cheerleader. You score honestly on a 1-5 scale where
3 means "competent but unremarkable", 4 means "notably good", and 5 is reserved
for writing you would genuinely want to keep. Most writing scores 2-3."""


def _build_review_prompt(rubric: dict, prompt_text: str, output_text: str) -> str:
    criteria_lines = []
    score_keys_example = []
    for c in rubric["criteria"]:
        desc = " ".join(c["description"].split())  # collapse whitespace
        criteria_lines.append(f"- {c['id']}: {desc}")
        score_keys_example.append(f'    "{c["id"]}": <integer 1-5>')
        
    criteria_block = "\n".join(criteria_lines)
    scores_block_example = ",\n".join(score_keys_example)

    return f"""I will show you a writing prompt and a piece of writing produced in response.
Score the writing against each rubric criterion on a 1-5 integer scale.

RUBRIC:
{criteria_block}

PROMPT GIVEN TO THE WRITER:
\"\"\"
{prompt_text}
\"\"\"

WRITING TO SCORE:
\"\"\"
{output_text}
\"\"\"

Respond with ONLY a JSON object, no other text. Use exactly this format:

{{
  "scratchpad": "<think through the criteria and evaluate the writing here first>",
  "scores": {{
{scores_block_example}
  }},
  "notes": "<one or two sentences naming the single most interesting thing about this piece and the single biggest weakness>"
}}"""


def _short_hash(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _rubric_hash(rubric: dict) -> str:
    # Hash just the criteria section — reviewer model changes shouldn't
    # invalidate a rubric version.
    blob = yaml.safe_dump(rubric["criteria"], sort_keys=True)
    return _short_hash(blob)


def _load_prompts(gen_rows: list[dict]) -> dict[str, str]:
    """Re-read prompt files so the reviewer sees the actual prompt, not a hash."""
    prompts_dir = ROOT / "prompts"
    needed = {row["prompt_id"] for row in gen_rows}
    out = {}
    for pid in needed:
        path = prompts_dir / f"{pid}.txt"
        if not path.exists():
            sys.exit(f"ERROR: prompt file missing: {path}")
        out[pid] = path.read_text(encoding="utf-8")
    return out


def _unsanitize_output(text: str) -> str:
    """Reverse explore.py's TSV sanitization (best-effort)."""
    return text.replace("⏎", "\n")


def _load_generations() -> list[dict]:
    if not GEN_PATH.exists():
        sys.exit(f"ERROR: {GEN_PATH} not found. Run explore.py first.")
    with GEN_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_existing_keys() -> set[tuple[str, int]]:
    if not REV_PATH.exists():
        return set()
    keys = set()
    with REV_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            keys.add((row["run_id"], int(row["sample_idx"])))
    return keys


def _open_append_writer():
    new_file = not REV_PATH.exists()
    f = REV_PATH.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=REV_FIELDS, delimiter="\t")
    if new_file:
        writer.writeheader()
        f.flush()
    return f, writer


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of the reviewer's response.

    Models sometimes wrap JSON in ```json fences or add a sentence of preamble
    despite the instruction not to. This finds the first {...} block and
    parses it. Raises ValueError on failure.
    """
    # Strip code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find first balanced JSON object. Cheap approach: find first { and last }.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in reviewer response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _validate_scores(obj: dict, rubric: dict) -> tuple[dict, str]:
    """Ensure scores cover every criterion and are 1-5 integers."""
    if "scores" not in obj or not isinstance(obj["scores"], dict):
        raise ValueError("response missing 'scores' object")
    scores = obj["scores"]
    expected = {c["id"] for c in rubric["criteria"]}
    missing = expected - set(scores.keys())
    if missing:
        raise ValueError(f"missing scores for: {sorted(missing)}")
    cleaned = {}
    for k, v in scores.items():
        if k not in expected:
            continue  # ignore extra keys
        if not isinstance(v, (int, float)) or not (1 <= v <= 5):
            raise ValueError(f"score for {k!r} is not 1-5: {v!r}")
        cleaned[k] = int(v)
    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)
    return cleaned, notes


def _salvage_response(text: str, rubric: dict) -> tuple[dict, str]:
    """Fallback regex extraction to salvage scores and notes from broken JSON."""
    scores = {}
    expected = {c["id"] for c in rubric["criteria"]}
    
    for k in expected:
        # Match either quoted or unquoted keys like: "concreteness": 4 or concreteness: 4
        pattern = rf'"{k}"\s*:\s*([1-5])|\b{k}\b\s*:\s*([1-5])'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # group 1 is quoted match, group 2 is unquoted match
            val = match.group(1) or match.group(2)
            scores[k] = int(val)
            
    missing = expected - set(scores.keys())
    if missing:
        raise ValueError(f"Regex fallback missing scores for: {sorted(missing)}")
        
    # Attempt to pull out something resembling notes
    notes_match = re.search(r'"notes"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if notes_match:
        notes = notes_match.group(1)
    else:
        # Just grab raw snippets as notes
        notes = "[Salvaged via Regex] " + text.replace("\n", " ").replace("\t", " ").strip()[:300]
        
    return scores, notes


def main() -> None:
    rubric = yaml.safe_load(RUBRIC.read_text(encoding="utf-8"))
    reviewer = rubric["reviewer"]
    rhash = _rubric_hash(rubric)

    gens = _load_generations()
    # Only review successful generations — no point scoring error rows.
    gens = [g for g in gens if g["status"] == "ok" and g["output_text"].strip()]

    prompts = _load_prompts(gens)
    done = _load_existing_keys()

    todo = [g for g in gens if (g["run_id"], int(g["sample_idx"])) not in done]
    print(f"Generations eligible for review: {len(gens)}.")
    print(f"Already reviewed: {len(done)}. To review now: {len(todo)}.")
    if not todo:
        print("Nothing to do.")
        return

    f, writer = _open_append_writer()
    try:
        for gen in todo:
            prompt_text = prompts[gen["prompt_id"]]
            output_text = _unsanitize_output(gen["output_text"])
            review_prompt = _build_review_prompt(rubric, prompt_text, output_text)

            record = {
                "run_id": gen["run_id"],
                "sample_idx": gen["sample_idx"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reviewer_provider": reviewer["provider"],
                "reviewer_model": reviewer["model"],
                "rubric_hash": rhash,
                "status": "ok",
                "error": "",
                "scores_json": "",
                "notes": "",
            }

            MAX_RETRIES = 3
            for attempt in range(MAX_RETRIES):
                try:
                    result = generate(
                        provider=reviewer["provider"],
                        model=reviewer["model"],
                        prompt=review_prompt,
                        T=float(reviewer["temperature"]),
                        top_p=float(reviewer["top_p"]),
                        top_k=reviewer.get("top_k"),
                        max_tokens=int(reviewer["max_tokens"]),
                        system=REVIEW_SYSTEM,
                    )
                    try:
                        parsed = _extract_json(result.text)
                        scores, notes = _validate_scores(parsed, rubric)
                        record["scores_json"] = json.dumps(scores, sort_keys=True)
                        record["notes"] = notes.replace("\t", " ").replace("\n", " ")
                        record["status"] = "ok"
                        break  # Success! Exit the retry loop
                    except (ValueError, json.JSONDecodeError) as e:
                        # Attempt a regex fallback
                        try:
                            scores, notes = _salvage_response(result.text, rubric)
                            record["status"] = "ok"
                            record["scores_json"] = json.dumps(scores, sort_keys=True)
                            record["notes"] = notes.replace("\t", " ").replace("\n", " ")
                            break  # Success regex salvage! Exit the retry loop
                        except ValueError as fallback_e:
                            if attempt == MAX_RETRIES - 1:
                                record["status"] = "parse_error"
                                record["error"] = f"{type(e).__name__}: {e} (Fallback regex failed: {fallback_e})"
                                record["notes"] = result.text[:500].replace("\t", " ").replace("\n", " ")
                            else:
                                print(f"  ↻ Retrying {gen['run_id']}/{gen['sample_idx']} (parse failed on attempt {attempt + 1})")
                                continue  # Try again

                except UnsupportedParam as e:
                    record["status"] = "error"
                    record["error"] = f"UnsupportedParam: {e}"
                    break  # Don't retry API config errors
                except ProviderConfigError as e:
                    record["status"] = "error"
                    record["error"] = f"ProviderConfigError: {e}"
                    break
                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"{type(e).__name__}: {e}"
                    # We could retry generic connection exceptions here if we wanted to
                    break

            writer.writerow(record)
            f.flush()

            marker = "✓" if record["status"] == "ok" else "✗"
            preview = record["scores_json"] if record["status"] == "ok" else record["error"][:60]
            print(f"  {marker} {gen['run_id']}/{gen['sample_idx']}  {preview}")
    finally:
        f.close()

    print()
    print(f"Done. Reviews in {REV_PATH}.")


if __name__ == "__main__":
    main()
