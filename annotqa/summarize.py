from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def distribution_bucket(score: float) -> str:
    if score <= 49:
        return "0-49"
    if score <= 69:
        return "50-69"
    if score <= 84:
        return "70-84"
    return "85-100"


def summarize_output(
    input_jsonl: Path,
    criterion_max_points: Optional[Dict[str, int]] = None,
    score_anchors: Optional[List[Dict[str, str]]] = None,
) -> Dict:
    overall: List[float] = []
    feedback_counter: Counter[str] = Counter()
    rationale_counter: Counter[str] = Counter()
    criterion_scores: Dict[str, List[float]] = {}
    criterion_max = criterion_max_points or {}
    anchors = score_anchors or [
        {"range": "0-39", "label": "Poor"},
        {"range": "40-59", "label": "Needs improvement"},
        {"range": "60-74", "label": "Fair"},
        {"range": "75-89", "label": "Good"},
        {"range": "90-100", "label": "Excellent"},
    ]

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            results = row.get("results", {})
            if results.get("flags", {}).get("parse_error"):
                continue
            for fb in results.get("feedback", []):
                msg = fb.get("message", "")
                if msg:
                    feedback_counter[msg] += 1
            rubric_points = results.get("rubric_points", {})
            by_criterion = rubric_points.get("by_criterion", {})
            for criterion_id, points in by_criterion.items():
                criterion_scores.setdefault(criterion_id, []).append(float(points))
            rationale_by_criterion = rubric_points.get("rationale_by_criterion", {})
            for criterion_id, rationale in rationale_by_criterion.items():
                if rationale:
                    rationale_counter[f"{criterion_id}: {rationale}"] += 1

    section_scores_0_100: Dict[str, float] = {}
    for criterion_id, points in sorted(criterion_scores.items()):
        avg_points = statistics.fmean(points) if points else 0.0
        max_points = float(criterion_max.get(criterion_id, max(points) if points else 1.0))
        if max_points <= 0:
            section_scores_0_100[criterion_id] = 0.0
        else:
            section_scores_0_100[criterion_id] = max(0.0, min(100.0, 100.0 * (avg_points / max_points)))

    overall_final_0_100 = (
        statistics.fmean(list(section_scores_0_100.values())) if section_scores_0_100 else 0.0
    )
    overall.append(overall_final_0_100)

    top_feedback = [f"{msg} ({cnt})" for msg, cnt in feedback_counter.most_common(8)]
    top_rationale = [f"{msg} ({cnt})" for msg, cnt in rationale_counter.most_common(8)]
    feedback_summary = "; ".join(top_feedback) if top_feedback else "No feedback generated."
    reasoning_summary = "; ".join(top_rationale) if top_rationale else "No rationale generated."

    report = {
        "final_score_0_100": overall_final_0_100,
        "section_scores_0_100": section_scores_0_100,
        "llm_summary": {
            "feedback": feedback_summary,
            "reasoning": reasoning_summary,
        },
        "score_anchors": anchors,
    }
    return report


def write_report(report: Dict, out_json: Path, out_csv: Path | None = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["final_score_0_100", report["final_score_0_100"]])
            for section_id, value in sorted(report.get("section_scores_0_100", {}).items()):
                writer.writerow([f"section_score_0_100.{section_id}", value])
            writer.writerow(["llm_summary.feedback", report.get("llm_summary", {}).get("feedback", "")])
            writer.writerow(["llm_summary.reasoning", report.get("llm_summary", {}).get("reasoning", "")])
            for anchor in report.get("score_anchors", []):
                writer.writerow([f"score_anchor.{anchor.get('range', '')}", anchor.get("label", "")])
