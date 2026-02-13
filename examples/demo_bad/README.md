# Demo Data

This directory contains a generated rubric and JSONL dataset.

## Commands

```bash
annotqa validate --input demo.jsonl --rubric rubric_v1_abtie.yaml
annotqa run --input demo.jsonl --rubric rubric_v1_abtie.yaml --out out.jsonl --dry_run
annotqa summarize --input out.jsonl --out report.json
```
