# annotqa

`annotqa` is a Python 3.11+ CLI for annotation QA with:
- grammar correction (LLM)
- rubric scoring by criterion (LLM using anchors)
- deterministic scoring aggregation and export

It is built so new annotation formats can be supported by adding a new rubric YAML (no code changes).

## 1) Setup

### Requirements
- Python `3.11+`
- OpenAI API key for non-dry-run execution

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure API key
```bash
export OPENAI_API_KEY="sk-..."
```

Optional persistent setup (`zsh`):
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

## 2) Supported Inputs

`--input` supports:
- a `.jsonl` file (streamed line-by-line)
- a `.json` file (object or array)
- a directory (recursive scan of `.jsonl` + `.json`)

## 3) Run Commands

### Validate (offline, no LLM)
```bash
annotqa validate --input <path> --rubric <rubric.yaml>
```

- prints parse/validation counts
- exit code `2` if parse errors exist, else `0`

### Run (main pipeline)
```bash
annotqa run \
  --input <path> \
  --rubric <rubric.yaml> \
  --out <dir>/out.jsonl
```

Options:
- `--model` default `gpt-4.1-mini`
- `--concurrency` default `4`
- `--batch_tokens` default `3000`
- `--max_retries` default `1`
- `--batch_timeout_s` default `120` (max time per LLM batch before marking LLM error and moving on)
- `--cache_db` default `.annotqa_cache.sqlite`
- `--dry_run` (no LLM calls)
- `--only_selected` (process only records where `isSelected == true`)

### Summarize (manual)
```bash
annotqa summarize \
  --input <dir>/out.jsonl \
  --out <dir>/report.json \
  --csv <dir>/summary.csv \
  --rubric <rubric.yaml>
```

Note: `annotqa run` already auto-generates `report.json` and `summary.csv`.

### Demo data generator
```bash
annotqa demo --out_dir examples/demo --n 2000 --seed 42 --error_rate 0.35
```

Writes:
- `examples/demo/rubric_v1_abtie.yaml`
- `examples/demo/demo.jsonl`
- `examples/demo/README.md`

### Run the generated demo data end-to-end
```bash
# 1) Generate data
annotqa demo --out_dir examples/demo_good --n 1000 --seed 42 --error_rate 0.10
annotqa demo --out_dir examples/demo_bad --n 1000 --seed 99 --error_rate 0.65

# 2) Validate (offline)
annotqa validate --input examples/demo_good/demo.jsonl --rubric examples/demo_good/rubric_v1_abtie.yaml
annotqa validate --input examples/demo_bad/demo.jsonl --rubric examples/demo_bad/rubric_v1_abtie.yaml

# 3) Run full pipeline
annotqa run --input examples/demo_good/demo.jsonl --rubric examples/demo_good/rubric_v1_abtie.yaml --out examples/demo_good/out.jsonl
annotqa run --input examples/demo_bad/demo.jsonl --rubric examples/demo_bad/rubric_v1_abtie.yaml --out examples/demo_bad/out.jsonl
```

After each `run`, inspect:
- `examples/demo_*/out.jsonl` (per-record detailed scoring + feedback)
- `examples/demo_*/final_out.json` (corrected merged dataset)
- `examples/demo_*/summary.csv` (final score + section scores + LLM summaries)
- `examples/demo_*/report.json` (JSON summary)

## 4) Outputs (from `annotqa run`)

Given `--out <dir>/out.jsonl`, the following are generated and overwritten on rerun:

- `<dir>/out.jsonl`  
  Per-input QA/audit output (one line per input record), including:
  - normalized fields
  - corrected text
  - rubric points and rationale per section
  - deterministic grammar/final scores
  - flags (`llm_error`, `parse_error`, etc.)

- `<dir>/final_out.json`  
  Full corrected records merged back into original shape (JSON array).

- `<dir>/report.json`  
  JSON summary.

- `<dir>/summary.csv`  
  Minimal score/summary export:
  - `final_score_0_100`
  - `section_score_0_100.<criterion_id>`
  - `llm_summary.feedback`
  - `llm_summary.reasoning`
  - score anchors

## 5) Add a New Rubric / Annotation Format

To support a new annotation JSON shape, create a new rubric YAML only.

### Rubric fields required
- `rubric_id`
- `version`
- `input_mapping`
  - `labels`: map logical label fields to selectors
  - `text`: map logical text fields to selectors
- `label_specs` (required/allowed values)
- `text_specs` (required/min_words/max_words)
- `criteria` (id, description, max_points, anchors)
- `scoring`

### Selector format
- `annotation.explanation`
- `$.annotation.explanation`

Rules:
- leading `$.` is optional
- keys are dot-traversed
- missing keys return `None`

### Example flow for a new dataset
```bash
annotqa validate --input data/new_set.jsonl --rubric rubrics/new_format.yaml
annotqa run --input data/new_set.jsonl --rubric rubrics/new_format.yaml --out outputs/new_set/out.jsonl
```

No code changes are needed if the rubric selectors/specs are correct.

## 6) Performance Tips

If runs are slow or rate-limited:
- reduce `--concurrency` (e.g. `2`)
- reduce `--batch_tokens` (e.g. `1500-2500`)
- reduce `--max_retries` (e.g. `0-1`)
- reduce `--batch_timeout_s` (e.g. `45-90`) to avoid waiting too long on stuck batches

Example:
```bash
annotqa run \
  --input examples/demo_good/demo.jsonl \
  --rubric examples/demo_good/rubric_v1_abtie.yaml \
  --out examples/demo_good/out.jsonl \
  --concurrency 2 \
  --batch_tokens 2000 \
  --max_retries 0 \
  --batch_timeout_s 60
```

## 7) Troubleshooting

- `OPENAI_API_KEY is required`  
  Export your key before non-dry-run execution.

- Scores are zero in `summary.csv`  
  Check `out.jsonl` for `results.flags.llm_error=true`.  
  If present, inspect `results.feedback` for API/format errors and rerun with lower concurrency.

- Need offline check only  
  Use `annotqa validate` and/or `annotqa run --dry_run`.

## 8) Development

Run tests:
```bash
pytest
```