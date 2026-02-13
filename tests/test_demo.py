from pathlib import Path

from annotqa.demo import generate_demo


def test_demo_deterministic(tmp_path: Path):
    out1 = tmp_path / "d1"
    out2 = tmp_path / "d2"
    generate_demo(out1, n=50, seed=42, error_rate=0.35)
    generate_demo(out2, n=50, seed=42, error_rate=0.35)

    data1 = (out1 / "demo.jsonl").read_text(encoding="utf-8")
    data2 = (out2 / "demo.jsonl").read_text(encoding="utf-8")
    rubric1 = (out1 / "rubric_v1_abtie.yaml").read_text(encoding="utf-8")
    rubric2 = (out2 / "rubric_v1_abtie.yaml").read_text(encoding="utf-8")

    assert data1 == data2
    assert rubric1 == rubric2
