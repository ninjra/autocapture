"""Table extractor plugin fixture for tests."""

from __future__ import annotations

from typing import Any

from autocapture.enrichment.table_extractor import (
    TableColumn,
    TableExtractionResult,
    TableExtractionWindow,
    TableExtractorSpec,
    TableSchema,
    UpsertPolicy,
)


class _BadValue:
    pass


def _default_schema() -> TableSchema:
    return TableSchema(
        name="extracted_table",
        columns=[
            TableColumn(name="id", dtype="text", nullable=False),
            TableColumn(name="note", dtype="text", nullable=True),
            TableColumn(name="note_embedding", dtype="blob", nullable=True),
        ],
        primary_key=["id"],
    )


def _default_upsert() -> UpsertPolicy:
    return UpsertPolicy(mode="upsert", conflict_columns=["id"])


def _coerce_rows(raw_rows: Any) -> list[dict[str, object]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, object]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        converted: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, dict) and value.get("__bad__"):
                converted[key] = _BadValue()
            else:
                converted[key] = value
        rows.append(converted)
    return rows


class TestTableExtractor:
    def __init__(self, spec: TableExtractorSpec, rows: list[dict[str, object]]) -> None:
        self._spec = spec
        self._rows = rows

    def describe(self) -> TableExtractorSpec:
        return self._spec

    def extract(self, request):
        return TableExtractionResult(table=request.table, rows=self._rows, upsert=self._spec.upsert)


def create_extractor(context, **_kwargs):
    settings = context.plugin_settings if isinstance(context.plugin_settings, dict) else {}
    table_payload = settings.get("table")
    if isinstance(table_payload, dict):
        table = TableSchema.model_validate(table_payload)
    else:
        table = _default_schema()

    window_payload = settings.get("window")
    if isinstance(window_payload, dict):
        window = TableExtractionWindow.model_validate(window_payload)
    else:
        window = TableExtractionWindow(lookback_days=1)

    upsert_payload = settings.get("upsert")
    if isinstance(upsert_payload, dict):
        upsert = UpsertPolicy.model_validate(upsert_payload)
    else:
        upsert = _default_upsert()

    prompt_template = settings.get("prompt_template") or "Extract rows."
    rows = _coerce_rows(settings.get("rows"))
    spec = TableExtractorSpec(
        table=table,
        window=window,
        prompt_template=str(prompt_template),
        upsert=upsert,
    )
    return TestTableExtractor(spec, rows)
