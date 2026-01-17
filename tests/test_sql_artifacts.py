from __future__ import annotations

from autocapture.enrichment.sql_artifacts import extract_sql_artifacts


def test_extract_sql_artifacts_tables_and_order() -> None:
    text = (
        "```sql\n"
        "SELECT * FROM sales.orders;\n"
        "UPDATE inventory.stock SET qty = 1 WHERE sku = 'A';\n"
        "```"
    )
    artifacts = extract_sql_artifacts(text, [])
    assert len(artifacts.sql_statements) == 2
    assert artifacts.sql_statements[0].operation == "select"
    assert artifacts.sql_statements[1].operation == "update"
    tables_first = " ".join(artifacts.sql_statements[0].tables)
    tables_second = " ".join(artifacts.sql_statements[1].tables)
    assert "sales" in tables_first or "orders" in tables_first
    assert "inventory" in tables_second or "stock" in tables_second
    assert "operation: select" in artifacts.artifact_text


def test_extract_sql_artifacts_parse_errors() -> None:
    text = "SELECT FROM"
    artifacts = extract_sql_artifacts(text, [])
    assert artifacts.sql_statements
    assert artifacts.sql_statements[0].parse_error is not None
