from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import yaml


def _good_explanation(choice: str) -> str:
    other = "B" if choice == "A" else "A"
    return (
        f"I prefer {choice} because its composition is cleaner and the style matches the prompt better than {other}. "
        f"It also feels more consistent with the requested tone."
    )


def _bad_grammar(text: str, rng: random.Random) -> str:
    variants = [
        text.lower(),
        text.replace(".", ""),
        text.replace("better", "bettr"),
        "  " + text + "  ",
        text.replace(". ", " and "),
    ]
    return rng.choice(variants)


def _rubric_violation(text: str, rng: random.Random, labels: Dict[str, str]) -> tuple[str | None, Dict[str, str]]:
    mode = rng.choice(["too_short", "too_long", "vague", "no_compare", "contradiction", "missing"])
    if mode == "too_short":
        return "Looks better.", labels
    if mode == "too_long":
        sentence = "This is somewhat okay but also not okay and I keep repeating this thought."
        return " ".join([sentence] * 30), labels
    if mode == "vague":
        return "It looks better and I like it more.", labels
    if mode == "no_compare":
        return "I picked this one because it seems nice and generally strong.", labels
    if mode == "contradiction":
        flipped = dict(labels)
        flipped["aestheticPreference"] = "B" if labels["aestheticPreference"] == "A" else "A"
        return text, flipped
    # missing
    return None, labels


def _random_label(rng: random.Random) -> str:
    return rng.choice(["A", "B", "TIE"])


def _invalid_label(rng: random.Random) -> str:
    return rng.choice(["C", "UNKNOWN", ""])


def build_demo_rubric() -> Dict:
    return {
        "rubric_id": "rubric_v1_abtie",
        "version": "1.0.0",
        "input_mapping": {
            "labels": {
                "aestheticPreference": "annotation.aestheticPreference",
                "fitsDomain": "annotation.fitsDomain",
                "fitsPrompt": "annotation.fitsPrompt",
                "fitsTone": "annotation.fitsTone",
            },
            "text": {
                "explanation": "annotation.explanation",
            },
        },
        "label_specs": {
            "aestheticPreference": {"selector": "annotation.aestheticPreference", "required": True, "allowed": ["A", "B", "TIE"]},
            "fitsDomain": {"selector": "annotation.fitsDomain", "required": True, "allowed": ["A", "B", "TIE"]},
            "fitsPrompt": {"selector": "annotation.fitsPrompt", "required": True, "allowed": ["A", "B", "TIE"]},
            "fitsTone": {"selector": "annotation.fitsTone", "required": True, "allowed": ["A", "B", "TIE"]},
        },
        "text_specs": {
            "explanation": {"selector": "annotation.explanation", "required": True, "min_words": 12, "max_words": 140}
        },
        "criteria": [
            {
                "id": "evidence_specificity",
                "description": "Explanation cites concrete details of the preferred option.",
                "max_points": 5,
                "anchors": {"0": "No concrete evidence.", "3": "Some specific detail.", "5": "Multiple precise supporting details."},
            },
            {
                "id": "comparative_reasoning",
                "description": "Explanation compares A and B with a clear rationale.",
                "max_points": 5,
                "anchors": {"0": "No comparison.", "3": "Partial comparison.", "5": "Clear and complete comparison."},
            },
            {
                "id": "alignment_with_labels",
                "description": "Explanation aligns with declared labels and does not contradict them.",
                "max_points": 5,
                "anchors": {"0": "Contradictory.", "3": "Mostly aligned.", "5": "Fully aligned."},
            },
            {
                "id": "clarity_structure",
                "description": "Writing is readable, coherent, and well-structured.",
                "max_points": 5,
                "anchors": {"0": "Unclear.", "3": "Partially clear.", "5": "Very clear and structured."},
            },
            {
                "id": "rubric_completeness",
                "description": "Covers requested dimensions: aesthetics, domain, prompt, and tone.",
                "max_points": 5,
                "anchors": {"0": "Misses most dimensions.", "3": "Covers some dimensions.", "5": "Covers all dimensions."},
            },
        ],
        "scoring": {
            "grammar": {"penalty_scale": 300, "penalty_cap": 30},
            "final": {"method": "weighted_sum", "rubric_weight": 0.7, "grammar_weight": 0.3},
        },
    }


def generate_demo(out_dir: Path, n: int, seed: int, error_rate: float) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rubric_path = out_dir / "rubric_v1_abtie.yaml"
    data_path = out_dir / "demo.jsonl"
    readme_path = out_dir / "README.md"

    rubric = build_demo_rubric()
    rubric_path.write_text(yaml.safe_dump(rubric, sort_keys=False), encoding="utf-8")

    with data_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            pref = _random_label(rng)
            labels = {
                "aestheticPreference": pref,
                "fitsDomain": _random_label(rng),
                "fitsPrompt": _random_label(rng),
                "fitsTone": _random_label(rng),
            }
            explanation = _good_explanation("A" if pref == "TIE" else pref)

            if rng.random() < error_rate:
                if rng.random() < 0.55:
                    explanation = _bad_grammar(explanation, rng)
                else:
                    explanation, labels = _rubric_violation(explanation, rng, labels)

            if rng.random() < error_rate * 0.12:
                bad_key = rng.choice(list(labels.keys()))
                labels[bad_key] = _invalid_label(rng)

            obj = {
                "_id": f"demo-{i}",
                "taskId": f"task-{i % 25}",
                "userId": f"user-{i % 13}",
                "annotation": {
                    "aestheticPreference": labels["aestheticPreference"],
                    "fitsDomain": labels["fitsDomain"],
                    "fitsPrompt": labels["fitsPrompt"],
                    "fitsTone": labels["fitsTone"],
                    "explanation": explanation,
                },
            }
            if rng.random() < error_rate * 0.08:
                # Occasionally remove required field.
                obj["annotation"].pop(rng.choice(["fitsPrompt", "explanation"]))
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    readme = (
        "# Demo Data\n\n"
        "This directory contains a generated rubric and JSONL dataset.\n\n"
        "## Commands\n\n"
        "```bash\n"
        "annotqa validate --input demo.jsonl --rubric rubric_v1_abtie.yaml\n"
        "annotqa run --input demo.jsonl --rubric rubric_v1_abtie.yaml --out out.jsonl --dry_run\n"
        "annotqa summarize --input out.jsonl --out report.json\n"
        "```\n"
    )
    readme_path.write_text(readme, encoding="utf-8")
