from annotqa.scoring import (
    clamp,
    compute_grammar_metrics,
    compute_rubric_score,
    levenshtein_distance,
)


def test_levenshtein_distance():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "abc") == 0


def test_grammar_penalty_math():
    metrics, grammar_score = compute_grammar_metrics(
        {"explanation": "abcde"},
        {"explanation": "abXde"},
        penalty_scale=300,
        penalty_cap=30,
    )
    # dist=1, denom=5 -> edit_fraction=0.2, penalty=min(30, 60)=30, score=70
    assert round(metrics.edit_fraction, 6) == 0.2
    assert round(metrics.penalty_points, 6) == 30.0
    assert round(grammar_score, 6) == 70.0


def test_rubric_score():
    assert compute_rubric_score(20, 25) == 80.0
    assert clamp(0, 100, 999) == 100
