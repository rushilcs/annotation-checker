PRD — Annotation QA Tool (Bulk Grammar + Rubric Conformity + Hybrid Scoring)

0) TL;DR
Build a production-grade CLI tool called “annotqa” that ingests annotation JSON files (primarily JSONL, streaming), normalizes arbitrary annotation formats via a pluggable rubric YAML config, calls an LLM to (1) grammar-correct free-text fields and (2) evaluate rubric conformity using anchored scoring (non-deterministic), then computes the final quantitative score deterministically from:
- LLM rubric points (anchored)
- Measured grammar edit magnitude (edit fraction / percent)

Per-record output must include: corrected text, structured feedback, rubric points + rationale, grammar metrics, deterministic final score, and flags. Adding a new annotation format must require only adding a new rubric YAML file (no code changes).

1) Goals

1.1 Functional goals
Given:
- input annotations (JSONL / JSON array / directory)
- a rubric YAML
Produce:
- corrected text (grammar-only, preserve meaning)
- rubric conformity evaluation (anchored scoring by LLM)
- deterministic hybrid score
- JSONL outputs (1 output per 1 input)
- optional summary report

1.2 Must-have capabilities
- Streaming JSONL ingest (line-by-line, no full-file load)
- Directory recursion ingest (.jsonl, .json)
- Pluggable rubrics (selectors + criteria in YAML)
- Deterministic grammar penalty scoring
- Deterministic final score computation
- Async batching + concurrency
- Retry + exponential backoff
- SQLite caching
- Demo data generator (creates rubric + mixed-quality dataset)
- Works offline for validate + dry_run (no API key needed)

2) Non-goals
- Building a web app (CLI-first)
- Judging “correctness” of A vs B unless rubric explicitly requires it
- Training a model

3) Data Inputs

3.1 Supported input formats
- JSONL (preferred): each line is a JSON object, streamed
- JSON array file: list of objects (may be loaded depending on size; prefer streaming where possible)
- Directory: recursively ingest .jsonl/.json

3.2 Example input shape (user-provided)
Root metadata like _id, taskId, userId, timestamps; nested annotation object with:
- labels: aestheticPreference, fitsDomain, fitsPrompt, fitsTone (A/B/TIE)
- free-text: explanation

3.3 Output record (authoritative)
Each output line corresponds to exactly one input record. Never drop records.

High-level output shape:
- input_meta: source_file, line_number (for JSONL), extracted id fields, and optionally raw id fields
- rubric: rubric_id, rubric_version
- normalized: extracted labels + text
- llm: model, latency_ms, prompt_hash
- results:
  - corrected_text (per text field)
  - rubric_points (LLM)
  - grammar_metrics (deterministic)
  - final_scores (deterministic)
  - feedback (structured list)
  - flags (parse_error, llm_error, missing_required_fields, invalid_label_values, low_confidence)

4) Rubric System (Plug-in)

4.1 Rubric YAML responsibilities
A rubric YAML defines:
- How to extract fields (selectors) from arbitrary JSON shapes
- Label constraints (allowed values, required)
- Text constraints (required, min/max word counts)
- Rubric criteria with score anchors (0..N) and max_points
- Scoring config (grammar penalty mapping + final score method/weights)

4.2 Selector design
Implement a simple dot-path selector:
- Accepts “annotation.explanation” and optionally “$.annotation.explanation”
- Traverses nested dict keys
- Missing key returns None
No JSONPath library; implement minimal selector logic in code.

4.3 Adding new annotation formats
To support a new input JSON shape, a user must only:
- Create a new rubric YAML with updated selectors under input_mapping
- Adjust criteria/anchors if needed
No code changes.

5) LLM Evaluation Design (Anchored, Non-deterministic)
LLM must do rubric conformity and grammar correction.

Hard rules to enforce in prompting:
- Grammar correction must preserve meaning
- Must not flip A/B/TIE or modify label values
- Must not add new factual claims (only clarity/grammar)
- Must output structured JSON matching a schema

Anchored scoring:
- Each criterion has anchors (example: 0, 3, 5) describing what qualifies
- LLM must assign integer points per criterion (0..max_points)
- LLM must provide short rationale per criterion

6) Hybrid Scoring (Deterministic Final Score)

6.1 Grammar penalty (deterministic)
Compute edit distance between original and corrected text:
- levenshtein_distance(original, corrected)
- edit_fraction = dist / max(len(original), len(corrected), 1)

Convert to penalty points:
- penalty_points = min(penalty_cap, penalty_scale * edit_fraction)
Defaults:
- penalty_scale = 300
- penalty_cap = 30

Compute grammar score:
- grammar_0_100 = clamp(0, 100, 100 - penalty_points)

Store metrics per record:
- original_chars, corrected_chars
- levenshtein_distance
- edit_fraction
- penalty_points
If multiple text fields exist, aggregate by char-weighted average.

6.2 Rubric score (deterministic scaling of LLM points)
LLM returns:
- total rubric points
- max_total rubric points
Compute:
- rubric_0_100 = 100 * (total / max_total)

6.3 Final score (deterministic)
Configured by rubric YAML. Default method:
- overall_0_100 = clamp(0, 100, rubric_weight * rubric_0_100 + grammar_weight * grammar_0_100)
Default weights:
- rubric_weight = 0.70
- grammar_weight = 0.30

7) Processing Pipeline (Per Record)

Step 1: Parse JSON object
- If parse fails, write output line with parse_error flag and continue

Step 2: Normalize via rubric mapping
- Extract labels + text using selectors
- Normalize whitespace in text fields

Step 3: Validate deterministically
- Missing required fields -> flags.missing_required_fields
- Invalid label values -> flags.invalid_label_values
- Word count violations -> feedback item + lower confidence suggestion
Validation must not crash the run.

Step 4: LLM (unless dry_run)
- Batch records by token estimate (chars/4)
- Include stable item_key per record for mapping
- Require JSON schema output

Step 5: Deterministic scoring
- Compute grammar metrics from orig vs corrected
- Compute rubric_0_100 from LLM points
- Compute final score deterministically

Step 6: Write output JSONL line immediately (streaming)

8) CLI UX

Commands:

A) annotqa validate
- Args: --input, --rubric
- Prints counts: parse errors, missing required fields, invalid labels, word count violations
- Exit code: 2 if any parse errors, else 0
- No API key required

B) annotqa run
- Args: --input, --rubric, --out
- Options:
  --model (default gpt-4.1-mini)
  --concurrency (default 8)
  --batch_tokens (default 6000)
  --max_retries (default 3)
  --cache_db (default .annotqa_cache.sqlite)
  --dry_run (no LLM; still writes output with flags)
  --only_selected (if isSelected exists; missing treated as false)
- Never crash on single bad record; always emit output lines

C) annotqa summarize
- Args: --input (output jsonl), --out (report json)
- Optional: --csv (summary.csv)
- Report:
  - count, mean/median/p10/p90 overall
  - stats for sub-scores
  - distribution buckets (0–49, 50–69, 70–84, 85–100)
  - top feedback messages by frequency
  - invalid/missing counts

D) annotqa demo (REQUIRED)
- Args: --out_dir, --n, --seed, --error_rate
- Writes:
  - demo rubric YAML
  - demo JSONL dataset
  - demo README explaining how to run validate/run/summarize
- Must generate mixed-quality data and scale to very large N via streaming writes.

9) Demo Dataset Requirements

The generator must produce:
- Clean examples:
  - correct labels
  - strong, specific, comparative explanations
  - good grammar

- Grammar-mistake examples:
  - lowercase starts, missing punctuation, typos, run-ons, extra spaces

- Rubric-violation examples:
  - too short / too long
  - vague (“looks better”)
  - no comparison between options
  - contradiction vs labels (“I chose A because B…”)
  - missing required fields occasionally
  - invalid label values occasionally (e.g., “C”)

Explanations should be short (1–3 sentences), but generation must support huge N (tons of data).

10) Performance + Reliability
- Streaming I/O
- Async batching
- Retry + exponential backoff
- SQLite cache keyed by hash of normalized content + rubric id/version + model
- Prompt hash recorded per run
- Never silently drop records

11) Acceptance Criteria
- validate works offline and returns correct exit codes
- run --dry_run works offline
- demo generates rubric + dataset + README and can scale to large N
- LLM rubric scoring uses anchors (not deterministic logic)
- Grammar penalty and final scoring are deterministic
- Adding a new rubric YAML supports a new JSON format without code changes