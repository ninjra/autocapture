# PromptOps

PromptOps automates prompt refreshes by ingesting sources, proposing prompt updates,
running evals, and optionally opening GitHub pull requests.

## How it works

1. **Fetch sources**
   - Configure sources in `promptops.sources` (HTTP(S) URLs or local file paths).
   - Snapshots are stored under `<data_dir>/promptops/sources/<run_id>/`.
   - Offline mode blocks network sources unless `privacy.cloud_enabled=true`.
2. **Propose prompt updates**
   - PromptOps reads prompt YAML files from `prompts/raw/`.
   - It calls the configured LLM provider (`routing.llm`) to generate proposed prompt YAML.
   - Versions are incremented automatically when required fields are present.
   - Derived prompts are compiled into `autocapture/prompts/derived/`.
3. **Run evals**
   - Uses `evals/golden_queries.json`.
   - Runs baseline vs proposed prompts and compares verifier pass rate with the
     configured tolerance (`promptops.acceptance_tolerance`).
4. **Create a PR or patch artifact**
   - If `promptops.github_token` and `promptops.github_repo` are set, PromptOps
     creates a branch and opens a PR.
   - Otherwise, a unified diff is written to `<data_dir>/promptops/patches/`.

## CLI

```bash
poetry run autocapture promptops run
poetry run autocapture promptops status
poetry run autocapture promptops list
```

## GitHub setup

Set the following in your config:

```yaml
promptops:
  enabled: true
  sources:
    - "https://example.com/notes"
    - "/path/to/local/source.txt"
  github_token: "ghp_..."
  github_repo: "org/repo"
  acceptance_tolerance: 0.02
```

## Safety

- PromptOps is disabled by default.
- Only curate trusted sources.
- Evals enforce a no-regression gate before PR creation.
