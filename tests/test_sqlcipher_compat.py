import importlib.util
from pathlib import Path

import pytest

from autocapture.security.sqlcipher import ensure_sqlcipher_create_function_compat


@pytest.mark.skipif(
    importlib.util.find_spec("pysqlcipher3") is None,
    reason="SQLCipher driver not installed",
)
def test_sqlcipher_create_function_accepts_deterministic(tmp_path: Path) -> None:
    import pysqlcipher3.dbapi2 as sqlcipher  # type: ignore

    ensure_sqlcipher_create_function_compat(sqlcipher)
    assert sqlcipher.sqlite_version_info <= (3, 8, 2)
    conn = sqlcipher.connect(str(tmp_path / "compat.db"))
    try:
        conn.create_function("autocapture_probe", 1, lambda x: x, deterministic=True)
    finally:
        conn.close()
