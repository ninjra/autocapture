# Agent Rules

## Scope & Priorities
- Keep changes focused, minimal, and aligned to the request.
- Prefer edits inside `autocapture/`, `tests/`, and `docs/`; avoid large restructures.
- Add tests when behavior changes; update docs when workflows change.
- Always use Poetry for Python projects.
- Always run `poetry install` before tests.
- Never use global pip.

## Local Commands
- Install deps: `poetry install --with dev` (Windows extras: `--extras "ui windows ocr ocr-gpu embed-fast"`).
- Lint: `poetry run ruff check .`
- Format: `poetry run black .`
- Tests: `poetry run pytest -q`
- Run: `poetry run autocapture`

## Code Style
- Python 3.12, Black/Ruff line length 100.
- Use snake_case for modules/functions, PascalCase for classes.
- Use explicit `pydantic` models at API/service boundaries.

## Project Map
- `autocapture/`: core capture pipeline, storage, API, UI, CLI entry points.
- `tests/`: pytest suites with shared fixtures in `tests/conftest.py`.
- `config/`: example config files.
- `docs/`: operational docs and dashboards.
- `alembic/` + `alembic.ini`: DB migrations.

## Safety & Hygiene
- Do not commit secrets or machine-specific paths.
- Keep config overrides out of git; use `config/example.yml` as reference.
