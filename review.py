"""
Read data/generations.tsv, score each generation against rubric.yaml,
append to data/reviews.tsv.

Separate from write.py so you can:
  - re-score with a different rubric without regenerating outputs
  - score with a different reviewer model
  - iterate on review criteria cheaply

Resumable the same way write.py is: rows already present are skipped.
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
SPEC = ROOT / "spec.yaml"

_spec_raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
_remote = str(_spec_raw.get("REMOTE_FOLDER_PATH") or "").strip()
REMOTE = Path(_remote) if _remote else None

DATA_DIR         = (REMOTE / "data")         if REMOTE else ROOT / "data"
REVIEW_FOLDER    = (REMOTE / "rubric")       if REMOTE else ROOT / "rubric"
TEMP_PROMPTS_DIR = (REMOTE / "temp_prompts") if REMOTE else ROOT / "temp_prompts"
OUTPUT_TEXT_DIR  = (REMOTE / "output_text")  if REMOTE else ROOT / "output_text"

RUBRIC   = REVIEW_FOLDER / "rubric.yaml"
GEN_PATH = DATA_DIR / "generations.tsv"
REV_PATH = DATA_DIR / "reviews.tsv"
TOP_PATH = OUTPUT_TEXT_DIR / "top_writing.tsv"

REV_FIELDS_BASE = [
    "run_id",
    "sample_idx",
    "timestamp",
    "reviewer_provider",
    "reviewer_model",
    "rubric_hash",       # so you can tell which rubric version produced these scores
    "status",            # ok | parse_error | error
    "error",
    "word_count",        # word count of the generated output
    # score columns are appended dynamically from rubric criteria in file order
    "notes",             # short free-text justification from the reviewer
]

REVIEW_SYSTEM = """You are a critical, unsentimental literary judge. 
Score each criterion on a 1-5 scale:
- 1 = deeply flawed, would not keep
- 2 = weak, needs major revision
- 3 = good, meets a professional standard (most writing)
- 4 = notably strong, rare, publishable with minor tweaks
- 5 = truly exceptional—among the best you have ever read; almost never award a 5.

Most writing should score 3. Only give a 4 if the work stands out in a professional context. Only give a 5 if you would publish it unchanged in a top literary magazine.

Be strict. Do not inflate scores. If in doubt, round down.
"""

TIEBREAKER_SYSTEM = """You are selecting the strongest writing samples from a tied set.
Use the provided score data and notes only. Return strict JSON with selected entries."""


def _build_review_prompt(
    rubric: dict,
    prompt_text: str,
    output_text: str,
    reference_text: str | None = None,
) -> str:
    criteria_lines = []
    score_keys_example = []
    for c in rubric["criteria"]:
        desc = " ".join(c["description"].split())  # collapse whitespace
        criteria_lines.append(f"- {c['id']}: {desc}")
        score_keys_example.append(f'    "{c["id"]}": <integer 1-5>')
        
    criteria_block = "\n".join(criteria_lines)
    scores_block_example = ",\n".join(score_keys_example)
    reference_block = ""
    if reference_text:
        reference_block = f"""

REFERENCE WRITING (for compare_to_reference):
\"\"\"
{reference_text}
\"\"\"
"""

    return f"""I will show you a writing prompt and a piece of writing produced in response.
Score the writing against each rubric criterion on a 1-5 integer scale where 1 is poor, 2 is good, 3 is very good, 4 is exceptional, and 5 is outstanding in every possible way.

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
{reference_block}

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


def _score_field_ids(rubric: dict) -> list[str]:
    """Return rubric criterion IDs in rubric.yaml order for TSV score columns."""
    return [c["id"] for c in rubric["criteria"]]


def _load_prompts(gen_rows: list[dict]) -> dict[str, str]:
    """Read combined prompt files from temp_prompts/ so the reviewer sees the actual prompt."""
    needed = {row["prompt_id"] for row in gen_rows}
    out = {}
    for pid in needed:
        path = TEMP_PROMPTS_DIR / f"{pid}.md"
        if not path.exists():
            sys.exit(f"ERROR: combined prompt file missing: {path}\nRun make_grid.py first to regenerate temp_prompts/.")
        out[pid] = path.read_text(encoding="utf-8")
    return out


def _load_reference_story(spec: dict) -> str | None:
    """Load optional reference writing from story_to_compare_path in spec.yaml."""
    story_path = (spec.get("story_to_compare_path") or "").strip()
    if not story_path:
        return None
    base = REMOTE if REMOTE else ROOT
    path = base / story_path
    if not path.exists():
        sys.exit(f"ERROR: story_to_compare_path file not found: {path}")
    return path.read_text(encoding="utf-8")


def _unsanitize_output(text: str) -> str:
    """Reverse write.py's TSV sanitization (best-effort)."""
    return text.replace("⏎", "\n")


def _strip_thinking(text: str) -> str:
    """Best-effort removal of chain-of-thought style scaffolding."""
    if not text:
        return text

    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

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


def _load_generation_output(gen_row: dict) -> str:
    """Load generation text from filename (.md) or fallback to legacy inline TSV text."""
    ref = (gen_row.get("output_text") or "").strip()
    if not ref:
        return ""

    if ref.lower().endswith(".md"):
        path = Path(ref)
        if not path.is_absolute():
            path = OUTPUT_TEXT_DIR / path
        if path.exists():
            return _strip_thinking(path.read_text(encoding="utf-8"))

    return _strip_thinking(_unsanitize_output(ref))


def _load_generations() -> list[dict]:
    if not GEN_PATH.exists():
        sys.exit(f"ERROR: {GEN_PATH} not found. Run write.py first.")
    with GEN_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_reviews() -> list[dict]:
    if not REV_PATH.exists():
        return []
    with REV_PATH.open(encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f, delimiter="\t"):
            normalized = {}
            for k, v in row.items():
                if k is None:
                    continue
                normalized[k.lstrip("\ufeff").strip()] = v
            rows.append(normalized)
        return rows


def _load_existing_keys() -> set[tuple[str, int]]:
    if not REV_PATH.exists():
        return set()
    keys = set()
    with REV_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            run_id = row.get("run_id") or row.get("\ufeffrun_id")
            sample_idx = row.get("sample_idx")
            if run_id is None or sample_idx is None:
                continue
            try:
                keys.add((run_id, int(sample_idx)))
            except ValueError:
                continue
    return keys


def _open_append_writer(rev_fields: list[str]):
    new_file = not REV_PATH.exists()

    if not new_file:
        file_text = REV_PATH.read_text(encoding="utf-8")

        if not file_text.strip():
            # Existing but empty file: treat like a new file and emit header.
            new_file = True
        else:
            lines = file_text.splitlines()
            first_line = lines[0] if lines else ""
            first_fields = next(csv.reader([first_line], delimiter="\t")) if first_line else []
            normalized_first = [f.lstrip("\ufeff").strip() for f in first_fields]

            if normalized_first != rev_fields:
                # If the first row looks like data rather than a header, repair in-place
                # by prepending the expected header once.
                looks_like_data_row = len(normalized_first) > 0 and normalized_first[0] != "run_id"
                if looks_like_data_row:
                    repaired = "\t".join(rev_fields) + "\n" + file_text
                    REV_PATH.write_text(repaired, encoding="utf-8")
                else:
                    sys.exit(
                        "ERROR: reviews.tsv header does not match current rubric-driven score columns.\n"
                        "Move/rename data/reviews.tsv (or start a new run_id) and run review.py again."
                    )

    f = REV_PATH.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=rev_fields, delimiter="\t")
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


def _parse_score_cell(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _build_ranked_candidates(reviews: list[dict], score_field_ids: list[str]) -> list[dict]:
    candidates = []
    overall_exists = "overall" in score_field_ids
    ai_isms_exists = "ai_isms" in score_field_ids

    for row in reviews:
        if row.get("status") != "ok":
            continue

        score_vals = {}
        for sid in score_field_ids:
            score_vals[sid] = _parse_score_cell(row.get(sid))

        base_fields = [sid for sid in score_field_ids if sid not in {"overall", "fit_to_prompt"}]
        base_nums = [score_vals[sid] for sid in base_fields if score_vals[sid] is not None]
        if not base_nums:
            base_nums = [score_vals[sid] for sid in score_field_ids if score_vals[sid] is not None]
        if not base_nums:
            continue

        base_score = sum(base_nums) / len(base_nums)
        overall_score = score_vals.get("overall") if overall_exists else None
        fit_to_prompt_score = score_vals.get("ai_isms") if ai_isms_exists else None

        candidates.append(
            {
                "run_id": row["run_id"],
                "sample_idx": int(row["sample_idx"]),
                "notes": row.get("notes", ""),
                "row": row,
                "scores": score_vals,
                "base_score": base_score,
                "overall_score": overall_score,
                "fit_to_prompt_score": fit_to_prompt_score,
            }
        )

    candidates.sort(
        key=lambda c: (
            c["base_score"],
            -999.0 if c["overall_score"] is None else c["overall_score"],
            -999.0 if c["fit_to_prompt_score"] is None else c["fit_to_prompt_score"],
        ),
        reverse=True,
    )
    return candidates


def _ai_break_tie(
    tied: list[dict],
    max_choices: int,
    reviewer: dict,
    score_field_ids: list[str],
) -> list[tuple[str, int]]:
    items = []
    for c in tied:
        score_parts = []
        for sid in score_field_ids:
            v = c["scores"].get(sid)
            if v is None:
                continue
            if float(v).is_integer():
                score_parts.append(f"{sid}={int(v)}")
            else:
                score_parts.append(f"{sid}={v}")
        items.append(
            {
                "run_id": c["run_id"],
                "sample_idx": c["sample_idx"],
                "base_score": round(c["base_score"], 4),
                "scores": ", ".join(score_parts),
                "notes": c["notes"],
            }
        )

    prompt = (
        "Choose the strongest entries from this tied set.\n"
        f"Maximum number of choices to return: {max_choices}.\n"
        "Return ONLY JSON in this format:\n"
        "{\n"
        "  \"selected\": [\n"
        "    {\"run_id\": \"...\", \"sample_idx\": <integer>}\n"
        "  ]\n"
        "}\n\n"
        "TIED CANDIDATES:\n"
        f"{json.dumps(items, ensure_ascii=True, indent=2)}"
    )

    try:
        result = generate(
            provider=reviewer["provider"],
            model=reviewer["model"],
            prompt=prompt,
            T=0.0,
            top_p=float(reviewer["top_p"]) if reviewer.get("top_p") is not None else None,
            top_k=reviewer.get("top_k"),
            max_tokens=int(reviewer["max_tokens"]) if reviewer.get("max_tokens") is not None else None,
            system=TIEBREAKER_SYSTEM,
        )
        obj = _extract_json(result.text)
        selected = obj.get("selected", [])
        out = []
        for s in selected:
            rid = str(s.get("run_id", "")).strip()
            sidx = s.get("sample_idx")
            if not rid or not isinstance(sidx, (int, float)):
                continue
            out.append((rid, int(sidx)))
        if out:
            return out[:max_choices]
    except Exception:
        pass

    # Final deterministic fallback if model output is invalid.
    fallback = sorted((c["run_id"], c["sample_idx"]) for c in tied)
    return fallback[:max_choices]


def _select_top_candidates(
    candidates: list[dict],
    reviewer: dict,
    score_field_ids: list[str],
    limit: int = 3,
) -> list[dict]:
    selected = []
    i = 0
    while i < len(candidates) and len(selected) < limit:
        c = candidates[i]
        group_key = (c["base_score"], c["overall_score"], c["fit_to_prompt_score"])
        group = [c]
        j = i + 1
        while j < len(candidates):
            cj = candidates[j]
            if (cj["base_score"], cj["overall_score"], cj["fit_to_prompt_score"]) != group_key:
                break
            group.append(cj)
            j += 1

        slots = limit - len(selected)
        if len(group) <= slots:
            selected.extend(group)
        else:
            chosen_ids = set(_ai_break_tie(group, slots, reviewer, score_field_ids))
            chosen = [g for g in group if (g["run_id"], g["sample_idx"]) in chosen_ids]
            if len(chosen) < slots:
                leftovers = [g for g in group if (g["run_id"], g["sample_idx"]) not in chosen_ids]
                leftovers.sort(key=lambda x: (x["run_id"], x["sample_idx"]))
                chosen.extend(leftovers[: slots - len(chosen)])
            selected.extend(chosen[:slots])
            break

        i = j

    return selected[:limit]


def _write_top_writing(
    selected: list[dict],
    score_field_ids: list[str],
    gens: list[dict],
) -> None:
    OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    gen_map = {}
    for g in gens:
        gen_map[(g["run_id"], int(g["sample_idx"]))] = g

    fieldnames = [
        "rank",
        "run_id",
        "sample_idx",
        "base_score",
        "overall",
        "ai_isms",
        "word_count",
    ] + score_field_ids + [
        "prompt_id",
        "output_text",
        "notes",
    ]

    with TOP_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for idx, c in enumerate(selected, start=1):
            g = gen_map.get((c["run_id"], c["sample_idx"]), {})
            row = {
                "rank": idx,
                "run_id": c["run_id"],
                "sample_idx": c["sample_idx"],
                "base_score": f"{c['base_score']:.4f}",
                "overall": "" if c["overall_score"] is None else c["overall_score"],
                "ai_isms": "" if c["fit_to_prompt_score"] is None else c["fit_to_prompt_score"],
                "word_count": c["row"].get("word_count", ""),
                "prompt_id": g.get("prompt_id", ""),
                "output_text": g.get("output_text", ""),
                "notes": (c["notes"] or "").replace("\t", " ").replace("\n", " "),
            }
            for sid in score_field_ids:
                v = c["scores"].get(sid)
                row[sid] = "" if v is None else int(v) if float(v).is_integer() else v
            writer.writerow(row)


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
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    reviewer = rubric["reviewer"]
    rhash = _rubric_hash(rubric)
    score_field_ids = _score_field_ids(rubric)
    rev_fields = REV_FIELDS_BASE[:-1] + score_field_ids + ["notes"]
    reference_text = _load_reference_story(spec)

    active_criteria = rubric["criteria"]
    if reference_text is None:
        active_criteria = [c for c in active_criteria if c["id"] != "compare_to_reference"]
    score_rubric = {"criteria": active_criteria}

    gens = _load_generations()
    # Only review successful generations — no point scoring error rows.
    gens = [g for g in gens if g["status"] == "ok" and g["output_text"].strip()]

    prompts = _load_prompts(gens)
    done = _load_existing_keys()

    todo = [g for g in gens if (g["run_id"], int(g["sample_idx"])) not in done]
    print(f"Generations eligible for review: {len(gens)}.")
    print(f"Already reviewed: {len(done)}. To review now: {len(todo)}.")
    if todo:
        f, writer = _open_append_writer(rev_fields)
        try:
            for gen in todo:
                prompt_text = prompts[gen["prompt_id"]]
                output_text = _load_generation_output(gen)
                if not output_text.strip():
                    continue
                review_prompt = _build_review_prompt(
                    score_rubric,
                    prompt_text,
                    output_text,
                    reference_text=reference_text,
                )

                record = {
                    "run_id": gen["run_id"],
                    "sample_idx": gen["sample_idx"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "reviewer_provider": reviewer["provider"],
                    "reviewer_model": reviewer["model"],
                    "rubric_hash": rhash,
                    "status": "ok",
                    "error": "",
                    "word_count": len(output_text.split()),
                    "notes": "",
                }
                for sid in score_field_ids:
                    record[sid] = ""

                MAX_RETRIES = 3
                for attempt in range(MAX_RETRIES):
                    try:
                        result = generate(
                            provider=reviewer["provider"],
                            model=reviewer["model"],
                            prompt=review_prompt,
                            T=float(reviewer["temperature"]) if reviewer.get("temperature") is not None else None,
                            top_p=float(reviewer["top_p"]) if reviewer.get("top_p") is not None else None,
                            top_k=reviewer.get("top_k"),
                            max_tokens=int(reviewer["max_tokens"]) if reviewer.get("max_tokens") is not None else None,
                            system=REVIEW_SYSTEM,
                        )
                        try:
                            parsed = _extract_json(result.text)
                            scores, notes = _validate_scores(parsed, score_rubric)
                            if reference_text is None and any(c["id"] == "compare_to_reference" for c in rubric["criteria"]):
                                scores["compare_to_reference"] = None
                            for sid in score_field_ids:
                                val = scores.get(sid)
                                record[sid] = "" if val is None else int(val)
                            record["notes"] = notes.replace("\t", " ").replace("\n", " ")
                            record["status"] = "ok"
                            break  # Success! Exit the retry loop
                        except (ValueError, json.JSONDecodeError) as e:
                            # Attempt a regex fallback
                            try:
                                scores, notes = _salvage_response(result.text, score_rubric)
                                if reference_text is None and any(c["id"] == "compare_to_reference" for c in rubric["criteria"]):
                                    scores["compare_to_reference"] = None
                                record["status"] = "ok"
                                for sid in score_field_ids:
                                    val = scores.get(sid)
                                    record[sid] = "" if val is None else int(val)
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
                if record["status"] == "ok":
                    preview = ", ".join(
                        f"{sid}={record[sid]}" for sid in score_field_ids if record[sid] != ""
                    )
                else:
                    preview = record["error"][:60]
                print(f"  {marker} {gen['run_id']}/{gen['sample_idx']}  {preview}")
        finally:
            f.close()
    else:
        print("Nothing new to review.")

    reviews = _load_reviews()
    candidates = _build_ranked_candidates(reviews, score_field_ids)
    selected = _select_top_candidates(candidates, reviewer, score_field_ids, limit=3)
    _write_top_writing(selected, score_field_ids, gens)

    print()
    print(f"Done. Reviews in {REV_PATH}.")
    print(f"Top writing exported to {TOP_PATH} ({len(selected)} rows).")


if __name__ == "__main__":
    main()
