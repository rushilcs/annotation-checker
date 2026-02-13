from __future__ import annotations

from typing import Dict, Tuple

from annotqa.models import FinalScores, GrammarMetrics, RubricConfig


def clamp(low: float, high: float, value: float) -> float:
    return max(low, min(high, value))


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (ca != cb)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def compute_grammar_metrics(
    original_text: Dict[str, str | None],
    corrected_text: Dict[str, str | None],
    penalty_scale: float = 300.0,
    penalty_cap: float = 30.0,
) -> Tuple[GrammarMetrics, float]:
    total_weight = 0
    sum_edit_fraction = 0.0
    sum_penalty = 0.0
    total_original_chars = 0
    total_corrected_chars = 0
    total_distance = 0

    keys = set(original_text.keys()) | set(corrected_text.keys())
    for key in keys:
        orig = (original_text.get(key) or "").strip()
        corr = (corrected_text.get(key) or "").strip()
        dist = levenshtein_distance(orig, corr)
        denom = max(len(orig), len(corr), 1)
        edit_fraction = dist / denom
        penalty = min(penalty_cap, penalty_scale * edit_fraction)
        weight = denom
        total_weight += weight
        sum_edit_fraction += edit_fraction * weight
        sum_penalty += penalty * weight
        total_original_chars += len(orig)
        total_corrected_chars += len(corr)
        total_distance += dist

    if total_weight == 0:
        avg_edit_fraction = 0.0
        avg_penalty = 0.0
    else:
        avg_edit_fraction = sum_edit_fraction / total_weight
        avg_penalty = sum_penalty / total_weight

    grammar_0_100 = clamp(0, 100, 100 - avg_penalty)
    return (
        GrammarMetrics(
            original_chars=total_original_chars,
            corrected_chars=total_corrected_chars,
            levenshtein_distance=total_distance,
            edit_fraction=avg_edit_fraction,
            penalty_points=avg_penalty,
        ),
        grammar_0_100,
    )


def compute_rubric_score(rubric_total: int, rubric_max_total: int) -> float:
    if rubric_max_total <= 0:
        return 0.0
    return clamp(0, 100, 100.0 * (rubric_total / rubric_max_total))


def compute_final_scores(
    rubric_0_100: float,
    grammar_0_100: float,
    rubric: RubricConfig,
) -> FinalScores:
    final_cfg = rubric.scoring.final
    overall = clamp(
        0,
        100,
        final_cfg.rubric_weight * rubric_0_100 + final_cfg.grammar_weight * grammar_0_100,
    )
    return FinalScores(
        overall_0_100=overall,
        rubric_0_100=rubric_0_100,
        grammar_0_100=grammar_0_100,
    )
