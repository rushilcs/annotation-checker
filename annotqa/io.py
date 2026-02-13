from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List

from annotqa.models import InputMeta, ParsedInput


def iter_input_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}:
            yield path


def _extract_ids(obj: Dict) -> Dict:
    keys = ("_id", "id", "taskId", "userId")
    return {k: obj.get(k) for k in keys if k in obj}


def parse_inputs(input_path: Path) -> Generator[ParsedInput, None, None]:
    counter = 0
    for path in iter_input_files(input_path):
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line_no, raw_line in enumerate(f, 1):
                    counter += 1
                    item_key = f"{path}:{line_no}"
                    text = raw_line.strip()
                    if not text:
                        yield ParsedInput(
                            item_key=item_key,
                            input_meta=InputMeta(
                                source_file=str(path),
                                line_number=line_no,
                                extracted_ids={},
                            ),
                            parse_error="Empty line",
                            raw_line=raw_line.rstrip("\n"),
                        )
                        continue
                    try:
                        obj = json.loads(text)
                        if not isinstance(obj, dict):
                            raise ValueError("JSONL line is not an object")
                        yield ParsedInput(
                            item_key=item_key,
                            input_meta=InputMeta(
                                source_file=str(path),
                                line_number=line_no,
                                extracted_ids=_extract_ids(obj),
                            ),
                            raw=obj,
                        )
                    except Exception as exc:
                        yield ParsedInput(
                            item_key=item_key,
                            input_meta=InputMeta(
                                source_file=str(path),
                                line_number=line_no,
                                extracted_ids={},
                            ),
                            parse_error=str(exc),
                            raw_line=raw_line.rstrip("\n"),
                        )
        else:
            with path.open("r", encoding="utf-8") as f:
                try:
                    doc = json.load(f)
                except Exception as exc:
                    counter += 1
                    yield ParsedInput(
                        item_key=f"{path}:1",
                        input_meta=InputMeta(source_file=str(path), line_number=1, extracted_ids={}),
                        parse_error=str(exc),
                    )
                    continue

            if isinstance(doc, list):
                for idx, obj in enumerate(doc, 1):
                    counter += 1
                    if isinstance(obj, dict):
                        yield ParsedInput(
                            item_key=f"{path}:{idx}",
                            input_meta=InputMeta(
                                source_file=str(path), line_number=idx, extracted_ids=_extract_ids(obj)
                            ),
                            raw=obj,
                        )
                    else:
                        yield ParsedInput(
                            item_key=f"{path}:{idx}",
                            input_meta=InputMeta(source_file=str(path), line_number=idx, extracted_ids={}),
                            parse_error="JSON array item is not an object",
                        )
            elif isinstance(doc, dict):
                counter += 1
                yield ParsedInput(
                    item_key=f"{path}:1",
                    input_meta=InputMeta(source_file=str(path), line_number=1, extracted_ids=_extract_ids(doc)),
                    raw=doc,
                )
            else:
                counter += 1
                yield ParsedInput(
                    item_key=f"{path}:1",
                    input_meta=InputMeta(source_file=str(path), line_number=1, extracted_ids={}),
                    parse_error="JSON document is not an object or array",
                )


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
