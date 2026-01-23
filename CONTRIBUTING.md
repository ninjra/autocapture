# Contributing

Thanks for contributing to Autocapture. For setup and workflow basics, see `README.md`.

## AI assistance (model preference)

- Default to the highest reasoning/thinking model available for AI-assisted changes
  and reviews.
- If your account supports a "Pro Extended" tier/mode, prefer it for non-trivial work.
- When cost/latency requires a faster model, note that choice in the PR description and
  rerun any critical reasoning steps on the highest tier when possible.
- Keep model selection local (tool settings or environment variables); do not commit
  secrets or personal overrides.

## Testing discipline

- Always run `poetry install --with dev` before tests.
- Ensure tests pass before moving on to the next task or feature.
- This is mandatory for every change; do not proceed unless tests pass.

## Web console

- Source lives in `autocapture/ui/console` and builds into `autocapture/ui/web`.
- Build with:
  - `cd autocapture/ui/console`
  - `npm install`
  - `npm run build`
- Open the UI with `poetry run autocapture ui open`.
