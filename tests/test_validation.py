from pathlib import Path

from annotqa.models import InputMeta
from annotqa.rubric import load_rubric, prepare_item


def test_validation_flags_required_and_invalid():
    rubric = load_rubric(Path("rubrics/rubric_v1_abtie.yaml"))
    raw = {
        "_id": "x",
        "annotation": {
            "aestheticPreference": "C",
            "fitsDomain": "A",
            "fitsPrompt": "B",
            "fitsTone": "A",
            "explanation": "Too short.",
        },
    }
    prepared = prepare_item(raw, rubric, "k1", InputMeta(source_file="s.jsonl", line_number=1, extracted_ids={}))
    assert prepared.flags.invalid_label_values is True
    assert prepared.flags.missing_required_fields is False
    assert any(f.type == "word_count_violation" for f in prepared.feedback)


def test_validation_missing_required():
    rubric = load_rubric(Path("rubrics/rubric_v1_abtie.yaml"))
    raw = {"annotation": {"aestheticPreference": "A"}}
    prepared = prepare_item(raw, rubric, "k2", InputMeta(source_file="s.jsonl", line_number=2, extracted_ids={}))
    assert prepared.flags.missing_required_fields is True
