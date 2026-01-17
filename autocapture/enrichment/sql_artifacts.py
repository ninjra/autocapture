"""Deterministic SQL/code artifact extraction from screen text."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import sqlglot
from sqlglot import exp

from ..agents.schemas import CodeBlock, SqlStatement

SQL_ARTIFACTS_SCHEMA_VERSION = "v1"

_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\n(?P<body>.*?)```", re.DOTALL)
_SQL_KEYWORDS_RE = re.compile(r"(?i)\b(select|insert|update|delete|with|create|alter|drop|merge)\b")


@dataclass(frozen=True)
class SqlArtifacts:
    code_blocks: list[CodeBlock]
    sql_statements: list[SqlStatement]
    artifact_text: str

    def as_tags(self) -> dict:
        return {
            "schema_version": SQL_ARTIFACTS_SCHEMA_VERSION,
            "code_blocks": [block.model_dump() for block in self.code_blocks],
            "sql_statements": [stmt.model_dump() for stmt in self.sql_statements],
            "artifact_text": self.artifact_text,
        }


def extract_sql_artifacts(visible_text: str | None, regions: Iterable[dict] | None) -> SqlArtifacts:
    visible_text = visible_text or ""
    region_texts = [
        str(region.get("text_verbatim") or "")
        for region in (regions or [])
        if isinstance(region, dict)
    ]
    candidates = _collect_candidates([visible_text, *region_texts])
    code_blocks = _extract_code_blocks(candidates)
    sql_blocks = [block for block in code_blocks if _is_sql_block(block)]
    sql_statements: list[SqlStatement] = []
    for block in sql_blocks:
        sql_statements.extend(_parse_sql_statements(block.text))
    if not sql_statements:
        sql_statements.extend(_parse_sql_statements(_fallback_sql_text(visible_text)))
    artifact_text = _build_artifact_text(sql_statements)
    return SqlArtifacts(
        code_blocks=code_blocks,
        sql_statements=sql_statements,
        artifact_text=artifact_text,
    )


def _collect_candidates(texts: Iterable[str]) -> list[str]:
    output: list[str] = []
    for text in texts:
        raw = str(text or "")
        stripped = raw.strip()
        if stripped:
            output.append(stripped)
        normalized = " ".join(stripped.split())
        if normalized and normalized != stripped:
            output.append(normalized)
    return output


def _extract_code_blocks(texts: Iterable[str]) -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    for text in texts:
        for match in _CODE_BLOCK_RE.finditer(text):
            language = (match.group("lang") or "").strip().lower() or "text"
            body = (match.group("body") or "").strip()
            if not body:
                continue
            blocks.append(CodeBlock(language=language, text=body))
    return blocks


def _is_sql_block(block: CodeBlock) -> bool:
    if block.language in {"sql", "postgres", "postgresql", "mysql", "sqlite", "tsql"}:
        return True
    return _looks_like_sql(block.text)


def _looks_like_sql(text: str) -> bool:
    return bool(_SQL_KEYWORDS_RE.search(text or ""))


def _fallback_sql_text(text: str) -> str:
    if _looks_like_sql(text):
        return text
    return ""


def _parse_sql_statements(text: str) -> list[SqlStatement]:
    text = (text or "").strip()
    if not text:
        return []
    statements: list[SqlStatement] = []
    try:
        expressions = sqlglot.parse(text)
    except Exception as exc:
        statements.extend(_parse_sql_fallback(text, exc))
        return statements
    for expr in expressions:
        statements.append(_statement_from_expr(expr, None))
    return statements


def _parse_sql_fallback(text: str, exc: Exception) -> list[SqlStatement]:
    statements: list[SqlStatement] = []
    parts = [part.strip() for part in text.split(";") if part.strip()]
    for part in parts:
        try:
            expr = sqlglot.parse_one(part)
            statements.append(_statement_from_expr(expr, None))
        except Exception as inner_exc:
            statements.append(
                SqlStatement(
                    text=part,
                    operation="unknown",
                    tables=[],
                    parse_error=str(inner_exc),
                )
            )
    if not statements:
        statements.append(
            SqlStatement(
                text=text,
                operation="unknown",
                tables=[],
                parse_error=str(exc),
            )
        )
    return statements


def _statement_from_expr(expr: exp.Expression, parse_error: str | None) -> SqlStatement:
    operation = _operation_for_expr(expr)
    tables = _extract_tables(expr)
    try:
        text = expr.sql()
    except Exception:
        text = str(expr)
    return SqlStatement(
        text=text,
        operation=operation,
        tables=tables,
        parse_error=parse_error,
    )


def _operation_for_expr(expr: exp.Expression) -> str:
    if isinstance(expr, exp.Select):
        return "select"
    if isinstance(expr, exp.Insert):
        return "insert"
    if isinstance(expr, exp.Update):
        return "update"
    if isinstance(expr, exp.Delete):
        return "delete"
    if isinstance(expr, exp.Create):
        return "create"
    if isinstance(expr, exp.Alter):
        return "alter"
    if isinstance(expr, exp.Drop):
        return "drop"
    if isinstance(expr, exp.Merge):
        return "merge"
    key = getattr(expr, "key", None)
    return str(key or "unknown")


def _extract_tables(expr: exp.Expression) -> list[str]:
    tables: set[str] = set()
    for table in expr.find_all(exp.Table):
        name = table.sql()
        if name:
            tables.add(name)
    return sorted(tables)


def _build_artifact_text(statements: Iterable[SqlStatement]) -> str:
    parts: list[str] = []
    for stmt in statements:
        if stmt.text:
            parts.append(stmt.text)
        if stmt.tables:
            parts.append("tables: " + ", ".join(stmt.tables))
        if stmt.operation:
            parts.append(f"operation: {stmt.operation}")
    return " ".join(parts).strip()
