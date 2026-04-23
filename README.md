# explore

Temperature & sampling-parameter sweep harness for creative writing, inspired by
Karpathy's autoresearch pattern but adapted for inference-time experimentation
across multiple LLM providers.

## What this is

Three small scripts that each do one thing:

```
make_grid.py      reads a YAML spec, writes grid.tsv (the experiment plan)
explore.py        reads grid.tsv + prompts/, writes generations.tsv (raw samples)
review.py         reads generations.tsv + rubric.yaml, writes reviews.tsv (scores)
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
├── prompts/              one .txt file per prompt; filename (without .txt) = prompt_id
│   ├── p01_rewrite_sentence.txt
│   └── p02_open_paragraph.txt
└── data/                 generated files land here (gitignored)
    ├── grid.tsv
    ├── generations.tsv
    └── reviews.tsv
```

## Flow

1. Edit `spec.yaml` to describe the parameter grid.
2. `python make_grid.py` — produces `data/grid.tsv`.
3. Inspect grid.tsv. Count rows. Estimate cost. Decide whether to proceed.
4. `python explore.py` — generates samples, writes `data/generations.tsv`.
5. Edit `rubric.yaml` to describe what "good" means for your use case.
6. `python review.py` — scores generations, writes `data/reviews.tsv`.
7. Analyse reviews.tsv in pandas / a spreadsheet / whatever.

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
