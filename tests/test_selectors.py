from annotqa.selectors import extract_selector, set_selector_value


def test_extract_selector_basic():
    obj = {"annotation": {"explanation": "hello"}}
    assert extract_selector(obj, "annotation.explanation") == "hello"
    assert extract_selector(obj, "$.annotation.explanation") == "hello"


def test_extract_selector_missing():
    obj = {"annotation": {}}
    assert extract_selector(obj, "annotation.explanation") is None
    assert extract_selector(obj, "annotation.missing.value") is None


def test_set_selector_value():
    obj = {}
    set_selector_value(obj, "$.annotation.explanation", "fixed")
    assert obj == {"annotation": {"explanation": "fixed"}}
