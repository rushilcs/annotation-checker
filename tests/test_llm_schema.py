from annotqa.llm import _json_schema
from annotqa.models import LlmBatchResponse


def test_json_schema_clone_preserves_map_shapes():
    raw = LlmBatchResponse.model_json_schema()
    cloned = _json_schema(raw)
    assert isinstance(cloned, dict)
    # corrected_text is intentionally a map-like object in the contract.
    corrected_text = (
        cloned.get("$defs", {})
        .get("LlmItemResult", {})
        .get("properties", {})
        .get("corrected_text", {})
    )
    assert corrected_text.get("type") == "object"
