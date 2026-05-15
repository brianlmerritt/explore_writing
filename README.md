# explore

Temperature & sampling-parameter sweep harness for creative writing, inspired by
Karpathy's autoresearch pattern but adapted for inference-time experimentation
across multiple LLM providers.

## What this is

Three small scripts that each do one thing:

```
make_grid.py      reads spec.yaml + recipe/prompt folders, writes data/grid.tsv
explore.py        reads data/grid.tsv, writes data/generations.tsv + output_text/*.md
review.py         reads generations + rubric.yaml, writes data/reviews.tsv
```

Everything is append-only and resumable. If `explore.py` dies partway through,
re-running skips rows already present in generations.tsv. Same for review.py.

## Directory layout

```
explore/
├── README.md
├── .env.example          copy to .env and fill in
├── .gitignore
├── requirements.txt
├── providers.py          thin adapter: OpenAI / Anthropic / Gemini / OpenRouter / local
├── make_grid.py
├── explore.py
├── review.py
├── spec.yaml             edit this to define your grid
├── rubric.yaml           edit this to define review criteria
├── prompt_examples/      local default prompt parts (.md)
├── prompt_recipe_examples/ local default recipes (.yaml)
├── temp_prompts/         generated combined prompts (from make_grid.py)
├── output_text/          generated outputs (.md) when REMOTE_FOLDER_PATH is not set
└── data/                 generated files land here (gitignored)
    ├── grid.tsv
    ├── generations.tsv
    └── reviews.tsv
```

## Flow

1. Edit `spec.yaml` to describe the parameter grid (recipes are recipe basenames, no .yaml).
2. `python make_grid.py` — produces `data/grid.tsv`.
3. Inspect grid.tsv. Count rows. Estimate cost. Decide whether to proceed.
4. `python explore.py` — generates samples, writes `data/generations.tsv`, and saves cleaned text to markdown files in the `output_text/` subfolder (under `REMOTE_FOLDER_PATH` if set, otherwise the local `output_text/` folder).
5. Edit `rubric.yaml` to describe what "good" means for your use case.
6. `python review.py` — scores generations, writes `data/reviews.tsv`.
7. Analyse reviews.tsv in pandas / a spreadsheet / whatever.

## Remote Folders

Set `REMOTE_FOLDER_PATH` in `spec.yaml` to point all scripts at a shared folder
(e.g. OneDrive, a mounted drive, or an SSH share). Leave it blank to use local defaults.

When `REMOTE_FOLDER_PATH` is set the scripts expect this subfolder structure:

```
<REMOTE_FOLDER_PATH>/
├── data/            generations.tsv, reviews.tsv, grid.tsv
├── prompts/         prompt part files (*.md)
├── prompt_recipes/  recipe files (*.yaml)
├── rubric/          rubric.yaml
├── temp_prompts/    expanded combined prompts (written by make_grid.py)
└── output_text/     generated outputs (*.md, written by explore.py)
```

When `REMOTE_FOLDER_PATH` is blank the local repo defaults are used:

| Subfolder      | Local default            |
|----------------|--------------------------|
| `data/`        | `data/`                  |
| `prompts/`     | `prompt_examples/`       |
| `prompt_recipes/` | `prompt_recipe_examples/` |
| `rubric/`      | project root (`.`)       |
| `temp_prompts/`| `temp_prompts/`          |
| `output_text/` | `output_text/`           |

Both absolute and relative paths are accepted. Example `spec.yaml` entry:

```yaml
REMOTE_FOLDER_PATH: "C:\\Users\\brian\\OneDrive\\Documents\\Writing\\Duo"
```

### Reference story

To use the `compare_to_reference` rubric criterion, set `story_to_compare_path` in
`spec.yaml`. The path can be absolute, or relative to `REMOTE_FOLDER_PATH` (or the
project root when `REMOTE_FOLDER_PATH` is not set):

```yaml
story_to_compare_path: "duo.md"                      # relative to REMOTE_FOLDER_PATH
# story_to_compare_path: "C:\\full\\path\\to\\duo.md"  # absolute
```

Leave it blank or omit it entirely to skip the criterion.

## Why pre-generate the grid

Reproducibility, resumability, and cost-awareness. See the grid file before you
spend the money. If run 47 of 200 fails, restart from row 48. Diff grid.tsv
across experiment versions to see what actually changed.

## Providers supported

| Provider    | top_k? | Notes                                           |
|-------------|--------|-------------------------------------------------|
| anthropic   | yes    | Native SDK                                      |
| openai      | no     | top_k silently dropped would be bad — we skip  |
| gemini      | yes    | Uses OpenAI-compatible endpoint                 |
| openrouter  | varies | Depends on upstream model; passes through      |
| local       | varies | OpenAI-compatible (Ollama, llama.cpp, vLLM)    |

A row whose provider doesn't support a requested parameter is logged with
`status=unsupported_param` rather than silently run with different settings.
