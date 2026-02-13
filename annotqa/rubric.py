from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from annotqa.models import FeedbackItem, Flags, NormalizedData, PreparedItem, RubricConfig
from annotqa.selectors import extract_selector


def load_rubric(path: Path) -> RubricConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RubricConfig.model_validate(raw)


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def normalize_record(raw: Dict, rubric: RubricConfig) -> NormalizedData:
    labels_map = rubric.input_mapping.get("labels", {})
    text_map = rubric.input_mapping.get("text", {})

    labels: Dict[str, Optional[str]] = {}
    for field, selector in labels_map.items():
        value = extract_selector(raw, selector)
        labels[field] = str(value) if value is not None else None

    text: Dict[str, Optional[str]] = {}
    for field, selector in text_map.items():
        value = extract_selector(raw, selector)
        if value is None:
            text[field] = None
        else:
            text[field] = normalize_whitespace(str(value))

    return NormalizedData(labels=labels, text=text)


def validate_normalized(normalized: NormalizedData, rubric: RubricConfig) -> Tuple[Flags, List[FeedbackItem]]:
    flags = Flags()
    feedback: List[FeedbackItem] = []

    for field, spec in rubric.label_specs.items():
        value = normalized.labels.get(field)
        if spec.required and (value is None or value == ""):
            flags.missing_required_fields = True
            feedback.append(
                FeedbackItem(
                    type="missing_required",
                    field=field,
                    message=f"Missing required label: {field}",
                    severity="error",
                )
            )
            continue
        if value is not None and spec.allowed and value not in spec.allowed:
            flags.invalid_label_values = True
            feedback.append(
                FeedbackItem(
                    type="invalid_label",
                    field=field,
                    message=f"Invalid label value '{value}' for {field}",
                    severity="error",
                )
            )

    for field, spec in rubric.text_specs.items():
        value = normalized.text.get(field)
        if spec.required and (value is None or value == ""):
            flags.missing_required_fields = True
            feedback.append(
                FeedbackItem(
                    type="missing_required",
                    field=field,
                    message=f"Missing required text: {field}",
                    severity="error",
                )
            )
            continue
        if value is None:
            continue
        wc = word_count(value)
        if wc < spec.min_words or wc > spec.max_words:
            feedback.append(
                FeedbackItem(
                    type="word_count_violation",
                    field=field,
                    message=f"Word count {wc} outside [{spec.min_words}, {spec.max_words}]",
                    severity="warning",
                )
            )
    return flags, feedback


def prepare_item(raw: Dict, rubric: RubricConfig, item_key: str, input_meta) -> PreparedItem:
    normalized = normalize_record(raw, rubric)
    flags, feedback = validate_normalized(normalized, rubric)
    return PreparedItem(
        item_key=item_key,
        input_meta=input_meta,
        raw=raw,
        normalized=normalized,
        flags=flags,
        feedback=feedback,
    )
