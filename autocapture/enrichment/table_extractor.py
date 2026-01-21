"""Table extraction plugin contract and storage pipeline."""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import text

from ..config import AppConfig
from ..indexing.sqlite_utils import vector_to_blob
from ..logging_utils import get_logger
from ..plugins import PluginManager
from ..plugins.errors import PluginPolicyError
from ..plugins.manifest import ExtensionManifestV1
from ..storage.database import DatabaseManager

_LOG = get_logger("enrichment.table_extractor")

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Embedder(Protocol):
    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]: ...


class TableColumn(BaseModel):
    name: str
    dtype: Literal["text", "integer", "real", "blob", "json"]
    nullable: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(f"Invalid column name: {value}")
        return value


class TableSchema(BaseModel):
    name: str
    columns: list[TableColumn]
    primary_key: list[str] = Field(default_factory=list)
    unique_keys: list[list[str]] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(f"Invalid table name: {value}")
        return value

    @field_validator("columns")
    @classmethod
    def _validate_columns(cls, value: list[TableColumn]) -> list[TableColumn]:
        names = [col.name for col in value]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate column names in table schema")
        return value

    @model_validator(mode="after")
    def _validate_constraints(self) -> "TableSchema":
        column_names = {col.name for col in self.columns}
        for key in self.primary_key:
            if key not in column_names:
                raise ValueError(f"Primary key column '{key}' missing from schema")
        for unique in self.unique_keys:
            for key in unique:
                if key not in column_names:
                    raise ValueError(f"Unique key column '{key}' missing from schema")
        return self


class TableExtractionWindow(BaseModel):
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    lookback_days: int | None = Field(None, ge=0)

    def resolved(self, now: dt.datetime) -> "TableExtractionWindow":
        if self.start or self.end:
            return self
        if self.lookback_days is None:
            return self
        start = now - dt.timedelta(days=int(self.lookback_days))
        return self.model_copy(update={"start": start, "end": now})


class TableExtractionSource(BaseModel):
    source_id: str
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class UpsertPolicy(BaseModel):
    mode: Literal["insert", "upsert", "ignore"] = "insert"
    conflict_columns: list[str] = Field(default_factory=list)
    update_columns: list[str] | None = None


class TableExtractorSpec(BaseModel):
    table: TableSchema
    window: TableExtractionWindow = Field(default_factory=TableExtractionWindow)
    prompt_template: str | None = None
    upsert: UpsertPolicy = Field(default_factory=UpsertPolicy)


class TableExtractionRequest(BaseModel):
    table: TableSchema
    window: TableExtractionWindow
    prompt_template: str | None = None
    sources: list[TableExtractionSource] = Field(default_factory=list)


class TableExtractionResult(BaseModel):
    table: TableSchema
    rows: list[dict[str, object]] = Field(default_factory=list)
    upsert: UpsertPolicy = Field(default_factory=UpsertPolicy)


class TableExtractor(Protocol):
    def describe(self) -> TableExtractorSpec: ...

    def extract(self, request: TableExtractionRequest) -> TableExtractionResult: ...


@dataclass(frozen=True)
class TableExtractionOutcome:
    status: str
    table: str | None = None
    inserted: int = 0
    reason: str | None = None


class StubTableExtractor:
    def __init__(self) -> None:
        self._spec = TableExtractorSpec(
            table=TableSchema(
                name="extracted_table",
                columns=[
                    TableColumn(name="id", dtype="text", nullable=False),
                    TableColumn(name="value", dtype="text", nullable=True),
                ],
                primary_key=["id"],
            ),
            window=TableExtractionWindow(lookback_days=1),
            prompt_template="",
            upsert=UpsertPolicy(mode="upsert", conflict_columns=["id"]),
        )

    def describe(self) -> TableExtractorSpec:
        return self._spec

    def extract(self, request: TableExtractionRequest) -> TableExtractionResult:
        return TableExtractionResult(table=request.table, rows=[], upsert=self._spec.upsert)


class TableExtractionService:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        *,
        embedder: Embedder | None = None,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        self._config = config
        self._db = db
        if embedder is None:
            from ..embeddings.service import EmbeddingService

            embedder = EmbeddingService(config.embed)
        self._embedder = embedder
        self._plugins = plugin_manager or PluginManager(config)

    def extract_and_store(
        self,
        request: TableExtractionRequest | None = None,
        *,
        now: dt.datetime | None = None,
    ) -> TableExtractionOutcome:
        if not self._config.table_extractor.enabled:
            return TableExtractionOutcome(status="disabled", reason="disabled")
        extractor_id = (self._config.routing.table_extractor or "disabled").strip().lower()
        if not extractor_id or extractor_id == "disabled":
            return TableExtractionOutcome(status="disabled", reason="routing_disabled")
        record = None
        try:
            record = self._plugins.resolve_record("table.extractor", extractor_id)
        except Exception as exc:
            _LOG.warning("Table extractor record lookup failed ({}): {}", extractor_id, exc)
            return TableExtractionOutcome(status="error", reason="extension_unavailable")
        if not self._config.table_extractor.allow_cloud and _requires_cloud(record.manifest):
            return TableExtractionOutcome(status="policy_blocked", reason="cloud_not_allowed")
        try:
            extractor = self._plugins.resolve_extension("table.extractor", extractor_id)
        except PluginPolicyError as exc:
            return TableExtractionOutcome(status="policy_blocked", reason=str(exc))
        except Exception as exc:
            _LOG.warning("Table extractor plugin failed ({}): {}", extractor_id, exc)
            return TableExtractionOutcome(status="error", reason="plugin_failed")
        spec = _describe_extractor(extractor)
        if request is None:
            if spec is None:
                raise RuntimeError("Table extractor did not provide a spec")
            request = TableExtractionRequest(
                table=spec.table,
                window=_resolve_window(spec.window, now),
                prompt_template=spec.prompt_template,
                sources=[],
            )
        else:
            request = request.model_copy(
                update={
                    "window": _resolve_window(request.window, now),
                    "prompt_template": request.prompt_template
                    or (spec.prompt_template if spec else None),
                }
            )
        result = TableExtractionResult.model_validate(extractor.extract(request))
        if not result.rows:
            return TableExtractionOutcome(status="ok", table=result.table.name, inserted=0)
        inserted = self._store_result(result)
        return TableExtractionOutcome(status="ok", table=result.table.name, inserted=inserted)

    def _store_result(self, result: TableExtractionResult) -> int:
        def _tx(session) -> int:
            self._ensure_table(session, result.table, result.upsert)
            inserted = 0
            for row in result.rows:
                values = self._prepare_row(row, result.table)
                self._insert_row(session, result.table, values, result.upsert)
                inserted += 1
            return inserted

        return self._db.transaction(_tx)

    def _ensure_table(
        self,
        session,
        schema: TableSchema,
        upsert: UpsertPolicy,
    ) -> None:
        if self._db.engine.dialect.name != "sqlite":
            raise RuntimeError("Table extractor storage only supports SQLite")
        column_sql = []
        for col in schema.columns:
            sql_type = _sql_type(col.dtype)
            nullable = "" if col.nullable else " NOT NULL"
            column_sql.append(f"{col.name} {sql_type}{nullable}")
        constraints = []
        if schema.primary_key:
            constraints.append(f"PRIMARY KEY ({', '.join(schema.primary_key)})")
        unique_sets = [set(keys) for keys in schema.unique_keys]
        conflict = upsert.conflict_columns or []
        if conflict:
            conflict_set = set(conflict)
            if schema.primary_key and conflict_set == set(schema.primary_key):
                pass
            elif conflict_set not in unique_sets:
                unique_sets.append(conflict_set)
        for keys in unique_sets:
            constraints.append(f"UNIQUE ({', '.join(sorted(keys))})")
        ddl_parts = column_sql + constraints
        ddl = f"CREATE TABLE IF NOT EXISTS {schema.name} ({', '.join(ddl_parts)})"
        session.execute(text(ddl))

    def _prepare_row(self, row: dict[str, object], schema: TableSchema) -> dict[str, object]:
        if not isinstance(row, dict):
            raise ValueError("Row must be a dict")
        values: dict[str, object] = {}
        for col in schema.columns:
            raw = row.get(col.name)
            if col.dtype == "json":
                raw = _coerce_json(raw)
            elif col.dtype == "integer":
                raw = None if raw is None else int(raw)
            elif col.dtype == "real":
                raw = None if raw is None else float(raw)
            elif col.dtype == "blob":
                raw = self._coerce_blob(raw, col.name, row)
            else:
                raw = None if raw is None else str(raw)
            if raw is None and not col.nullable:
                raise ValueError(f"Column '{col.name}' cannot be null")
            values[col.name] = raw
        return values

    def _coerce_blob(self, raw: object, col_name: str, row: dict[str, object]) -> object:
        if raw is None and col_name.endswith("_embedding"):
            base_key = col_name[: -len("_embedding")]
            base_value = row.get(base_key)
            if base_value is not None:
                vectors = self._embedder.embed_texts([str(base_value)])
                if vectors:
                    raw = vectors[0]
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, memoryview):
            return raw.tobytes()
        if isinstance(raw, (list, tuple)):
            return vector_to_blob([float(val) for val in raw])
        raise ValueError(f"Unsupported blob value for {col_name}")

    def _insert_row(
        self,
        session,
        schema: TableSchema,
        values: dict[str, object],
        upsert: UpsertPolicy,
    ) -> None:
        columns = [col.name for col in schema.columns]
        placeholders = ", ".join(f":{name}" for name in columns)
        base_sql = f"INSERT INTO {schema.name} ({', '.join(columns)}) VALUES ({placeholders})"
        mode = upsert.mode
        if mode == "ignore":
            sql = (
                f"INSERT OR IGNORE INTO {schema.name} "
                f"({', '.join(columns)}) VALUES ({placeholders})"
            )
        elif mode == "upsert":
            conflict = upsert.conflict_columns or schema.primary_key
            if not conflict:
                raise ValueError("Upsert policy requires conflict columns")
            update_cols = upsert.update_columns
            if update_cols is None:
                update_cols = [col for col in columns if col not in conflict]
            if update_cols:
                assignments = ", ".join(f"{col}=excluded.{col}" for col in update_cols)
                sql = f"{base_sql} ON CONFLICT({', '.join(conflict)}) DO UPDATE SET {assignments}"
            else:
                sql = f"{base_sql} ON CONFLICT({', '.join(conflict)}) DO NOTHING"
        else:
            sql = base_sql
        session.execute(text(sql), values)


def _sql_type(dtype: str) -> str:
    if dtype == "text":
        return "TEXT"
    if dtype == "integer":
        return "INTEGER"
    if dtype == "real":
        return "REAL"
    if dtype == "blob":
        return "BLOB"
    if dtype == "json":
        return "TEXT"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _coerce_json(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _resolve_window(
    window: TableExtractionWindow, now: dt.datetime | None
) -> TableExtractionWindow:
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return window.resolved(now)


def _requires_cloud(manifest: ExtensionManifestV1) -> bool:
    pillars = manifest.pillars
    if not pillars or not pillars.data_handling:
        return False
    handling = pillars.data_handling
    return handling.cloud == "required" or (
        handling.cloud == "optional" and not handling.supports_local
    )


def _describe_extractor(extractor: object) -> TableExtractorSpec | None:
    describe = getattr(extractor, "describe", None)
    if not callable(describe):
        return None
    spec = describe()
    try:
        return TableExtractorSpec.model_validate(spec)
    except Exception:
        return None
