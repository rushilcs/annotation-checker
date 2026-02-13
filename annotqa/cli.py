from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import typer

from annotqa.demo import generate_demo
from annotqa.io import parse_inputs
from annotqa.llm import LlmCache, LlmEvaluator, batch_items, make_cache_key
from annotqa.models import (
    FeedbackItem,
    Flags,
    ItemResult,
    LlmItemResult,
    LlmMeta,
    OutputRecord,
    PreparedItem,
    RubricPoints,
    ValidationSummary,
)
from annotqa.rubric import load_rubric, prepare_item
from annotqa.scoring import compute_final_scores, compute_grammar_metrics, compute_rubric_score
from annotqa.selectors import set_selector_value
from annotqa.summarize import summarize_output, write_report

app = typer.Typer(help="Annotation QA tool")


def _item_result_from_llm(
    prepared: PreparedItem,
    llm_item: LlmItemResult | None,
    rubric,
    llm_error: bool,
    llm_error_message: str | None = None,
) -> ItemResult:
    if llm_item is None:
        corrected = {k: (v or "") for k, v in prepared.normalized.text.items()}
        rubric_total = 0
        rubric_max = sum(c.max_points for c in rubric.criteria)
        by_criterion = {c.id: 0 for c in rubric.criteria}
        rationale = {c.id: "" for c in rubric.criteria}
        llm_feedback: List[FeedbackItem] = []
        low_conf = False
    else:
        corrected = {k: llm_item.corrected_text.get(k, prepared.normalized.text.get(k) or "") for k in prepared.normalized.text}
        rubric_total = llm_item.rubric_points.total
        rubric_max = llm_item.rubric_points.max_total
        by_criterion = dict(llm_item.rubric_points.by_criterion)
        rationale = dict(llm_item.rubric_points.rationale_by_criterion)
        llm_feedback = [
            FeedbackItem(type=f.type, field=f.field, message=f.message, severity=f.severity)
            for f in llm_item.feedback
        ]
        low_conf = llm_item.low_confidence or llm_item.confidence < 0.5

    grammar_cfg = rubric.scoring.grammar
    grammar_metrics, grammar_0_100 = compute_grammar_metrics(
        prepared.normalized.text,
        corrected,
        penalty_scale=grammar_cfg.penalty_scale,
        penalty_cap=grammar_cfg.penalty_cap,
    )
    rubric_0_100 = compute_rubric_score(rubric_total, rubric_max)
    final_scores = compute_final_scores(rubric_0_100, grammar_0_100, rubric)

    flags = Flags(
        parse_error=False,
        llm_error=llm_error,
        missing_required_fields=prepared.flags.missing_required_fields,
        invalid_label_values=prepared.flags.invalid_label_values,
        low_confidence=low_conf,
    )
    feedback_items = [*prepared.feedback, *llm_feedback]
    if llm_error and llm_error_message:
        feedback_items.append(
            FeedbackItem(
                type="llm_error",
                field=None,
                message=llm_error_message[:500],
                severity="error",
            )
        )

    return ItemResult(
        corrected_text=corrected,
        rubric_points=RubricPoints(
            by_criterion=by_criterion,
            total=rubric_total,
            max_total=rubric_max,
            rationale_by_criterion=rationale,
        ),
        grammar_metrics=grammar_metrics,
        final_scores=final_scores,
        feedback=feedback_items,
        flags=flags,
    )


def _build_parse_error_output(parsed, rubric) -> OutputRecord:
    feedback = []
    if parsed.parse_error:
        feedback.append(
            FeedbackItem(
                type="parse_error",
                field=None,
                message=parsed.parse_error,
                severity="error",
            )
        )
    return OutputRecord(
        input_meta=parsed.input_meta,
        rubric={"rubric_id": rubric.rubric_id, "rubric_version": rubric.version},
        normalized={"labels": {}, "text": {}},
        llm=LlmMeta(model=None, latency_ms=None, prompt_hash=None),
        results=ItemResult(
            corrected_text={},
            rubric_points=RubricPoints(by_criterion={}, total=0, max_total=0, rationale_by_criterion={}),
            feedback=feedback,
            flags=Flags(parse_error=True),
        ),
    )


def _write_output_line(out_file, record: OutputRecord) -> None:
    out_file.write(record.model_dump_json() + "\n")


def _write_final_json_record(final_file, record: Dict, is_first: bool) -> bool:
    if not is_first:
        final_file.write(",\n")
    final_file.write(json.dumps(record, ensure_ascii=False))
    return False


def _build_final_record_from_item(prepared: PreparedItem, item_result: ItemResult, rubric_cfg) -> Dict:
    merged = copy.deepcopy(prepared.raw)
    text_mapping = rubric_cfg.input_mapping.get("text", {})
    for field_name, corrected_value in item_result.corrected_text.items():
        selector = text_mapping.get(field_name)
        if selector:
            set_selector_value(merged, selector, corrected_value)
    merged["_annotqa"] = {
        "line_number": prepared.input_meta.line_number,
        "flags": item_result.flags.model_dump(),
    }
    return merged


def _build_final_record_from_parse_error(parsed) -> Dict:
    return {
        "_annotqa": {
            "source_file": parsed.input_meta.source_file,
            "line_number": parsed.input_meta.line_number,
            "parse_error": parsed.parse_error,
            "raw_line": parsed.raw_line,
            "flags": {"parse_error": True},
        }
    }


def _criterion_max_points_map(rubric_cfg) -> Dict[str, int]:
    return {criterion.id: criterion.max_points for criterion in rubric_cfg.criteria}


@app.command("validate")
def validate_cmd(
    input: Path = typer.Option(..., "--input"),
    rubric: Path = typer.Option(..., "--rubric"),
) -> None:
    rubric_cfg = load_rubric(rubric)
    summary = ValidationSummary()

    for parsed in parse_inputs(input):
        if parsed.parse_error:
            summary.parse_errors += 1
            continue
        prepared = prepare_item(parsed.raw or {}, rubric_cfg, parsed.item_key, parsed.input_meta)
        if prepared.flags.missing_required_fields:
            summary.missing_required_fields += 1
        if prepared.flags.invalid_label_values:
            summary.invalid_labels += 1
        summary.word_violations += sum(1 for fb in prepared.feedback if fb.type == "word_count_violation")

    typer.echo(f"parse_errors={summary.parse_errors}")
    typer.echo(f"missing_required_fields={summary.missing_required_fields}")
    typer.echo(f"invalid_labels={summary.invalid_labels}")
    typer.echo(f"word_violations={summary.word_violations}")

    raise typer.Exit(code=2 if summary.parse_errors > 0 else 0)


async def _process_batch(
    batch: Sequence[PreparedItem],
    rubric_cfg,
    model: str,
    dry_run: bool,
    llm: LlmEvaluator | None,
    cache: LlmCache,
    semaphore: asyncio.Semaphore,
    batch_timeout_s: int,
) -> List[Tuple[PreparedItem, ItemResult, LlmMeta]]:
    cached_results: Dict[str, LlmItemResult] = {}
    uncached: List[PreparedItem] = []
    for item in batch:
        key = make_cache_key(item, rubric_cfg, model)
        cached = cache.get(key)
        if cached:
            cached_results[item.item_key] = LlmItemResult.model_validate(cached)
        else:
            uncached.append(item)

    llm_by_key: Dict[str, LlmItemResult] = dict(cached_results)
    latency_ms = None
    prompt_hash = None
    llm_error_keys: set[str] = set()
    llm_error_message: str | None = None

    if not dry_run and uncached:
        assert llm is not None
        try:
            async with semaphore:
                result = await asyncio.wait_for(
                    llm.evaluate_batch(rubric_cfg, uncached),
                    timeout=batch_timeout_s,
                )
            latency_ms = result.latency_ms
            prompt_hash = result.prompt_hash
            llm_by_key.update(result.by_item_key)
            for item in uncached:
                llm_item = result.by_item_key.get(item.item_key)
                if llm_item:
                    cache_key = make_cache_key(item, rubric_cfg, model)
                    cache.set(cache_key, llm_item.model_dump(), model=model, prompt_hash=result.prompt_hash)
                else:
                    llm_error_keys.add(item.item_key)
        except Exception as exc:
            llm_error_message = str(exc)
            # Preserve per-item outputs with explicit LLM error flags.
            llm_error_keys.update(x.item_key for x in uncached)
    elif dry_run:
        llm_error_keys = set()

    output_rows: List[Tuple[PreparedItem, ItemResult, LlmMeta]] = []
    for item in batch:
        llm_item = llm_by_key.get(item.item_key)
        item_result = _item_result_from_llm(
            prepared=item,
            llm_item=llm_item if not dry_run else None,
            rubric=rubric_cfg,
            llm_error=(item.item_key in llm_error_keys),
            llm_error_message=llm_error_message if item.item_key in llm_error_keys else None,
        )
        output_rows.append(
            (
                item,
                item_result,
                LlmMeta(model=model if not dry_run else None, latency_ms=latency_ms, prompt_hash=prompt_hash),
            )
        )
    return output_rows


@app.command("run")
def run_cmd(
    input: Path = typer.Option(..., "--input"),
    rubric: Path = typer.Option(..., "--rubric"),
    out: Path = typer.Option(..., "--out"),
    model: str = typer.Option("gpt-4.1-mini", "--model"),
    concurrency: int = typer.Option(4, "--concurrency"),
    batch_tokens: int = typer.Option(3000, "--batch_tokens"),
    max_retries: int = typer.Option(1, "--max_retries"),
    cache_db: Path = typer.Option(Path(".annotqa_cache.sqlite"), "--cache_db"),
    dry_run: bool = typer.Option(False, "--dry_run"),
    only_selected: bool = typer.Option(False, "--only_selected"),
    batch_timeout_s: int = typer.Option(120, "--batch_timeout_s"),
) -> None:
    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        typer.echo("OPENAI_API_KEY is required for non-dry_run mode.")
        raise typer.Exit(code=1)
    rubric_cfg = load_rubric(rubric)
    out.parent.mkdir(parents=True, exist_ok=True)

    async def _runner() -> None:
        cache = LlmCache(cache_db)
        llm = None if dry_run else LlmEvaluator(model=model, max_retries=max_retries, cache=cache)
        semaphore = asyncio.Semaphore(concurrency)
        pending_items: List[PreparedItem] = []
        active: List[asyncio.Task] = []
        final_out = out.parent / "final_out.json"

        with out.open("w", encoding="utf-8") as out_file, final_out.open("w", encoding="utf-8") as final_file:
            is_first_final = True
            written_records = 0
            final_file.write("[\n")
            for parsed in parse_inputs(input):
                if parsed.parse_error:
                    _write_output_line(out_file, _build_parse_error_output(parsed, rubric_cfg))
                    is_first_final = _write_final_json_record(
                        final_file, _build_final_record_from_parse_error(parsed), is_first_final
                    )
                    continue
                raw = parsed.raw or {}
                if only_selected and not bool(raw.get("isSelected", False)):
                    continue
                prepared = prepare_item(raw, rubric_cfg, parsed.item_key, parsed.input_meta)
                pending_items.append(prepared)
                batches = batch_items(pending_items, batch_tokens=batch_tokens)
                if len(batches) > 1:
                    flush_batch = batches[0]
                    pending_items = batches[1]
                    task = asyncio.create_task(
                        _process_batch(
                            batch=flush_batch,
                            rubric_cfg=rubric_cfg,
                            model=model,
                            dry_run=dry_run,
                            llm=llm,
                            cache=cache,
                            semaphore=semaphore,
                            batch_timeout_s=batch_timeout_s,
                        )
                    )
                    active.append(task)
                    if len(active) >= concurrency:
                        done, pending = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                        active = list(pending)
                        for d in done:
                            for item, item_result, llm_meta in d.result():
                                record = OutputRecord(
                                    input_meta=item.input_meta,
                                    rubric={"rubric_id": rubric_cfg.rubric_id, "rubric_version": rubric_cfg.version},
                                    normalized=item.normalized,
                                    llm=llm_meta,
                                    results=item_result,
                                )
                                _write_output_line(out_file, record)
                                is_first_final = _write_final_json_record(
                                    final_file,
                                    _build_final_record_from_item(item, item_result, rubric_cfg),
                                    is_first_final,
                                )
                                written_records += 1
                                if written_records % 100 == 0:
                                    typer.echo(f"Progress: wrote {written_records} records...")

            if pending_items:
                active.append(
                    asyncio.create_task(
                        _process_batch(
                            batch=pending_items,
                            rubric_cfg=rubric_cfg,
                            model=model,
                            dry_run=dry_run,
                            llm=llm,
                            cache=cache,
                            semaphore=semaphore,
                            batch_timeout_s=batch_timeout_s,
                        )
                    )
                )
            for done in asyncio.as_completed(active):
                rows = await done
                for item, item_result, llm_meta in rows:
                    record = OutputRecord(
                        input_meta=item.input_meta,
                        rubric={"rubric_id": rubric_cfg.rubric_id, "rubric_version": rubric_cfg.version},
                        normalized=item.normalized,
                        llm=llm_meta,
                        results=item_result,
                    )
                    _write_output_line(out_file, record)
                    is_first_final = _write_final_json_record(
                        final_file,
                        _build_final_record_from_item(item, item_result, rubric_cfg),
                        is_first_final,
                    )
                    written_records += 1
                    if written_records % 100 == 0:
                        typer.echo(f"Progress: wrote {written_records} records...")
            final_file.write("\n]\n")
        cache.close()
        auto_report = out.parent / "report.json"
        auto_summary_csv = out.parent / "summary.csv"
        report = summarize_output(out, criterion_max_points=_criterion_max_points_map(rubric_cfg))
        write_report(report, out_json=auto_report, out_csv=auto_summary_csv)
        typer.echo(f"Wrote output JSONL to {out}")
        typer.echo(f"Wrote corrected JSON to {final_out}")
        typer.echo(f"Wrote report JSON to {auto_report}")
        typer.echo(f"Wrote summary CSV to {auto_summary_csv}")

    asyncio.run(_runner())


@app.command("summarize")
def summarize_cmd(
    input: Path = typer.Option(..., "--input"),
    out: Path = typer.Option(..., "--out"),
    rubric: Path | None = typer.Option(None, "--rubric"),
    csv: Path | None = typer.Option(None, "--csv"),
) -> None:
    criterion_max_points = None
    if rubric:
        rubric_cfg = load_rubric(rubric)
        criterion_max_points = _criterion_max_points_map(rubric_cfg)
    report = summarize_output(input, criterion_max_points=criterion_max_points)
    write_report(report, out_json=out, out_csv=csv)
    typer.echo(f"Wrote report to {out}")
    if csv:
        typer.echo(f"Wrote csv summary to {csv}")


@app.command("demo")
def demo_cmd(
    out_dir: Path = typer.Option(Path("examples/demo"), "--out_dir"),
    n: int = typer.Option(2000, "--n"),
    seed: int = typer.Option(42, "--seed"),
    error_rate: float = typer.Option(0.35, "--error_rate"),
) -> None:
    generate_demo(out_dir=out_dir, n=n, seed=seed, error_rate=error_rate)
    typer.echo(f"Generated demo files in {out_dir}")


if __name__ == "__main__":
    app()
