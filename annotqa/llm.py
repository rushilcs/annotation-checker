from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import AsyncOpenAI

from annotqa.models import LlmBatchResponse, LlmItemResult, PreparedItem, RubricConfig


def _stable_json(data: Dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_cache_key(item: PreparedItem, rubric: RubricConfig, model: str) -> str:
    payload = {
        "rubric_id": rubric.rubric_id,
        "rubric_version": rubric.version,
        "model": model,
        "labels": item.normalized.labels,
        "text": item.normalized.text,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def build_prompts(rubric: RubricConfig, items: Sequence[PreparedItem]) -> tuple[str, str, str]:
    criteria_lines: List[str] = []
    for c in rubric.criteria:
        anchors = ", ".join(f"{k}:{v}" for k, v in sorted(c.anchors.items(), key=lambda x: int(x[0])))
        criteria_lines.append(
            f"- {c.id} (max {c.max_points}): {c.description}. Anchors: {anchors}"
        )
    criteria_block = "\n".join(criteria_lines)

    system_prompt = (
        "You are an annotation QA evaluator.\n"
        "Rules:\n"
        "1) Correct grammar only and preserve meaning.\n"
        "2) Do not modify label values and do not flip A/B/TIE labels.\n"
        "3) Do not add new facts.\n"
        "4) Assign integer criterion points based on rubric anchors.\n"
        "5) Return valid JSON only matching the schema."
    )
    user_payload = {
        "rubric_id": rubric.rubric_id,
        "rubric_version": rubric.version,
        "criteria": [
            {
                "id": c.id,
                "description": c.description,
                "max_points": c.max_points,
                "anchors": c.anchors,
            }
            for c in rubric.criteria
        ],
        "items": [
            {
                "item_key": item.item_key,
                "labels": item.normalized.labels,
                "text": item.normalized.text,
            }
            for item in items
        ],
        "instructions": {
            "preserve_meaning": True,
            "grammar_only": True,
            "do_not_change_labels": True,
            "return_json_only": True,
        },
    }
    user_prompt = _stable_json(user_payload)
    prompt_hash_input = f"{system_prompt}\n{user_prompt}\n{criteria_block}"
    prompt_hash = hashlib.sha256(prompt_hash_input.encode("utf-8")).hexdigest()
    return system_prompt, user_prompt, prompt_hash


def estimate_item_tokens(item: PreparedItem) -> int:
    char_count = len(_stable_json({"labels": item.normalized.labels, "text": item.normalized.text}))
    return max(1, char_count // 4)


def _json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-cloned JSON schema for API transport."""
    return json.loads(json.dumps(schema))


def batch_items(items: Sequence[PreparedItem], batch_tokens: int) -> List[List[PreparedItem]]:
    batches: List[List[PreparedItem]] = []
    current: List[PreparedItem] = []
    current_tokens = 0
    for item in items:
        t = estimate_item_tokens(item)
        if current and current_tokens + t > batch_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


class LlmCache:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, cache_key: str) -> Dict | None:
        row = self.conn.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(self, cache_key: str, response_json: Dict, model: str, prompt_hash: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache(cache_key, response_json, model, prompt_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cache_key, _stable_json(response_json), model, prompt_hash, time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass
class LlmBatchResult:
    by_item_key: Dict[str, LlmItemResult]
    latency_ms: int
    prompt_hash: str


class LlmEvaluator:
    def __init__(self, model: str, max_retries: int, cache: LlmCache):
        self.model = model
        self.max_retries = max_retries
        self.cache = cache
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def evaluate_batch(self, rubric: RubricConfig, items: Sequence[PreparedItem]) -> LlmBatchResult:
        system_prompt, user_prompt, prompt_hash = build_prompts(rubric, items)
        schema = _json_schema(LlmBatchResponse.model_json_schema())
        start = time.perf_counter()

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_prompt}],
                        },
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "annotqa_batch",
                            "schema": schema,
                            "strict": False,
                        }
                    },
                )
                raw_text = self._extract_response_text(response)
                if not raw_text:
                    raise ValueError("Empty LLM response text")
                data = json.loads(raw_text)
                parsed = LlmBatchResponse.model_validate(data)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                return LlmBatchResult(
                    by_item_key={item.item_key: item for item in parsed.items},
                    latency_ms=elapsed_ms,
                    prompt_hash=prompt_hash,
                )
            except Exception as exc:
                last_error = exc
                # Only fallback for response-format/parse issues.
                if self._should_try_chat_fallback(exc):
                    try:
                        chat_response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "annotqa_batch",
                                    "schema": schema,
                                    "strict": False,
                                },
                            },
                        )
                        msg_content = chat_response.choices[0].message.content
                        if isinstance(msg_content, list):
                            raw_text = "".join(
                                part.get("text", "") for part in msg_content if isinstance(part, dict)
                            )
                        else:
                            raw_text = msg_content or ""
                        if not raw_text:
                            raise ValueError("Empty chat completion response text")
                        data = json.loads(raw_text)
                        parsed = LlmBatchResponse.model_validate(data)
                        elapsed_ms = int((time.perf_counter() - start) * 1000)
                        return LlmBatchResult(
                            by_item_key={item.item_key: item for item in parsed.items},
                            latency_ms=elapsed_ms,
                            prompt_hash=prompt_hash,
                        )
                    except Exception as fallback_exc:
                        last_error = RuntimeError(
                            f"Responses API error: {exc}; Chat Completions fallback error: {fallback_exc}"
                        )
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self._retry_delay_seconds(attempt))

        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        raw_text = getattr(response, "output_text", None)
        if raw_text:
            return raw_text
        text_parts: List[str] = []
        for out in getattr(response, "output", []):
            for chunk in getattr(out, "content", []):
                chunk_type = getattr(chunk, "type", "")
                if chunk_type in {"output_text", "text"}:
                    text_parts.append(getattr(chunk, "text", ""))
                elif isinstance(chunk, dict):
                    if chunk.get("type") in {"output_text", "text"}:
                        text_parts.append(chunk.get("text", ""))
        return "".join(text_parts)

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        # Smaller exponential backoff with jitter to reduce synchronized stalls.
        base = min(4.0, 0.75 * (2**attempt))
        return base + random.uniform(0.0, 0.25)

    @staticmethod
    def _should_try_chat_fallback(exc: Exception) -> bool:
        msg = str(exc).lower()
        # Do not double-call on rate limits/transient server issues.
        if "429" in msg or "rate limit" in msg or "timeout" in msg or "503" in msg or "502" in msg:
            return False
        # Fallback is useful for parse/format-related failures.
        return (
            "json" in msg
            or "schema" in msg
            or "empty llm response text" in msg
            or "validation" in msg
            or "output" in msg
        )
