from __future__ import annotations

from typing import Any


def extract_selector(data: Any, selector: str) -> Any:
    """Extract a value using minimal dot-path selectors.

    Supports:
    - "annotation.explanation"
    - "$.annotation.explanation"
    """
    if not selector:
        return None
    clean = selector[2:] if selector.startswith("$.") else selector
    parts = [p for p in clean.split(".") if p]
    current = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def set_selector_value(data: dict, selector: str, value: Any) -> None:
    """Set a value using minimal dot-path selectors.

    Missing intermediate keys are created as dicts.
    """
    if not selector:
        return
    clean = selector[2:] if selector.startswith("$.") else selector
    parts = [p for p in clean.split(".") if p]
    if not parts:
        return
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    if isinstance(current, dict):
        current[parts[-1]] = value
