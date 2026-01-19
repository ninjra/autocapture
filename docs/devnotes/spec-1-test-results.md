# SPEC-1 test results (2026-01-19)

## Environment
- `poetry install --with dev` (no dependency changes)

## Lint / format
- `poetry run ruff check .` → OK
- `poetry run black .` → OK

## Tests
- `poetry run pytest -q`
  - Result: `307 passed, 7 skipped, 1 warning`
  - Warning: `autocapture/plugins/hash.py` uses deprecated `importlib.abc.Traversable` (Python 3.14 deprecation warning).

## Re-run (2026-01-19)
- Re-ran after adding the entry-point plugin example; results unchanged.
