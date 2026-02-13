import json
from pathlib import Path

from annotqa.summarize import summarize_output, write_report


def test_summarize_includes_rubric_breakdown_and_rationale(tmp_path: Path):
    input_path = tmp_path / "out.jsonl"
    row = {
        "results": {
            "final_scores": {"overall_0_100": 82, "rubric_0_100": 80, "grammar_0_100": 86},
            "rubric_points": {
                "by_criterion": {"clarity_structure": 4, "comparative_reasoning": 3},
                "total": 7,
                "max_total": 10,
                "rationale_by_criterion": {
                    "clarity_structure": "Clear sentence structure and coherent flow.",
                    "comparative_reasoning": "Comparison is present but shallow.",
                },
            },
            "feedback": [{"message": "Good detail"}],
            "flags": {"parse_error": False, "llm_error": False, "missing_required_fields": False, "invalid_label_values": False, "low_confidence": False},
        }
    }
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = summarize_output(
        input_path,
        criterion_max_points={"clarity_structure": 5, "comparative_reasoning": 5},
    )
    assert "final_score_0_100" in report
    assert "section_scores_0_100" in report
    assert "clarity_structure" in report["section_scores_0_100"]
    assert report["llm_summary"]["reasoning"]

    out_json = tmp_path / "report.json"
    out_csv = tmp_path / "summary.csv"
    write_report(report, out_json=out_json, out_csv=out_csv)
    csv_text = out_csv.read_text(encoding="utf-8")
    assert "section_score_0_100.clarity_structure" in csv_text
    assert "llm_summary.reasoning" in csv_text
    assert "score_anchor.0-39" in csv_text
