# explore AI writing generator

Temperature & sampling-parameter sweep harness for creative writing, inspired by Karpathy's autoresearch pattern but adapted for inference-time experimentation across multiple LLM providers.  Also includes the ability to support multiple prompts using prompt recipes.

## What this is

Three small scripts that each do one thing:

```
make_grid.py      reads spec.yaml + recipe/prompt folders, writes data/grid.tsv for all combinations of ai models, temperature, prompts
write.py          reads data/grid.tsv, runs the AI model with prompt and temperature and writes data/generations.tsv + output_text/*.md (one output story or chapter per grid.tsv entry)
review.py         reads the AI output text + rubric.yaml, runs the AI review against the rubric for each generated story, outputs data/reviews.tsv with a review of each chapter, plus top_writing.tsv which has the top 3 stories (ai tie breaker if needed)
```

Everything is append-only and resumable. If `write.py` dies partway through, re-running skips rows already present in generations.tsv. Same for review.py.

## Directory layout

```
explore_writing/
├── README.md
├── .env.example          copy to .env and fill in
├── .gitignore
├── requirements.txt
├── providers.py          thin adapter: OpenAI / Anthropic / Gemini / OpenRouter / local
├── make_grid.py
├── write.py
├── review.py
├── spec.yaml             edit this to define your grid
├── rubric/rubric.yaml    edit this to define review criteria for local default use only
├── prompts/              local default prompt parts (.md)
├── prompt_recipes/       local default recipes (.yaml)
├── temp_prompts/         generated combined prompts (from make_grid.py)
├── output_text/          generated outputs (.md) when REMOTE_FOLDER_PATH is not set
└── data/                 generated files land here (gitignored)
    ├── grid.tsv
    ├── generations.tsv
    └── reviews.tsv
```

## Remote Folders

Set `REMOTE_FOLDER_PATH` in `spec.yaml` to point all scripts at an accessible shared folder. Leave it blank to use local defaults.

When `REMOTE_FOLDER_PATH` is set the scripts expect this subfolder structure:

```
<REMOTE_FOLDER_PATH>/
├── data/            generations.tsv, reviews.tsv, grid.tsv will be created here
├── prompts/         prompt part files (*.md)
├── prompt_recipes/  recipe files (*.yaml)
├── rubric/          rubric.yaml
├── temp_prompts/    expanded combined prompts (written by make_grid.py)
└── output_text/     generated outputs (*.md, written by write.py) plus top_writing.tsv (written by review.py) will go here
```

All you have to do to setup a scene / chapter remote auto write is:
1. copy rubric.yaml or create your own using that as a template and then:
2. create your prompts as markdown files e.g. characters.md, scene_beats1.md, scene_beats2.md, do_this_not_that.md or what ever you want
3. copy a prompt_recipe.yaml file and edit it with the prompts for that recipe e.g. combo1.yaml has characters, scene_beats1, combo 2 has characters, scene_beats2, do_this_not_that

When `REMOTE_FOLDER_PATH` is blank the local repo defaults are used with the same subfolder names - use that to test your AI connection

Both absolute and relative paths are accepted. Example `spec.yaml` entry:

```yaml
REMOTE_FOLDER_PATH: "C:\\Users\\brian\\OneDrive\\Documents\\Writing\\Duo"
```

## Flow

1. Edit `spec.yaml` to describe the parameter grid (recipes are recipe basenames, no .yaml).
2. Setup your remote folders as above, add the path to `spec.yaml`
3. `python make_grid.py` — produces `data/grid.tsv`.
4. Inspect grid.tsv. Count rows. Estimate cost. Decide whether to proceed.
5. `python write.py` — generates samples, writes `data/generations.tsv`, and saves cleaned text to markdown files in the `output_text/` subfolder (under `REMOTE_FOLDER_PATH` if set, otherwise the local `output_text/` folder).
6. Edit `rubric.yaml` to describe what "good" means for your use case.
7. Optional - if you have a reference version already set `story_to_compare_path` in `spec.yaml` and review will compare that also
8. `python review.py` — scores generations, writes `data/reviews.tsv` with all reviews and then saves the best 3 to `output_text/top_writing.tsv`
9. Analyse reviews.tsv and top_writing.tsv in pandas / a spreadsheet / whatever.  Use the run_id to link this to the prompt recipe, ai model and temperature in grid.tsv

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
| gemini      | no    | Uses OpenAI-compatible endpoint                 |
| openrouter  | varies | Depends on upstream model; passes through      |
| local       | varies | OpenAI-compatible (Ollama, llama.cpp, vLLM)    |

A row whose provider doesn't support a requested parameter is logged with
`status=unsupported_param` rather than silently run with different settings.

## To Do

- Create write_chapters.py to iterate over multiple folders to write a whole book
- write_chapters.py should work with or without reviews (just put one provider in `spec.yaml` to use just your favourite provider)
- write_chapters.py should be able to carry forward prompts and prompt recipes from one folder to the other, with a pause for you to edit
