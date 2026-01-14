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
   - Versions are incremented only when there is a semantic change to prompt content.
   - Derived prompts are compiled into `autocapture/prompts/derived/`.
3. **Run evals**
   - Uses `evals/golden_queries.json`.
   - Runs multiple baseline/proposed evals (`promptops.eval_repeats`) and aggregates
     via `promptops.eval_aggregation` (default: `worst_case`).
   - Applies a multi-metric gate: no-regression tolerances, floors/ceilings, and
     “must improve at least one metric” deltas.
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
  eval_repeats: 3
  eval_aggregation: "worst_case"
  require_improvement: true
  min_delta_verifier_pass_rate: 0.02
  min_delta_citation_coverage: 0.02
  min_delta_refusal_rate: 0.02
  min_delta_latency_ms: 200.0
  tolerance_citation_coverage: 0.02
  tolerance_refusal_rate: 0.02
  tolerance_latency_ms: 250.0
  min_verifier_pass_rate: 0.60
  min_citation_coverage: 0.60
  max_refusal_rate: 0.30
  max_mean_latency_ms: 15000.0
  max_source_bytes: 1048576
  max_source_excerpt_chars: 2000
  max_sources: 32
```

## Statuses

PromptOps records the run outcome as a status:

- `noop_no_semantic_change`: No prompt diffs beyond whitespace; evals skipped.
- `noop_no_improvement`: No metric improved by minimum delta; evals ran but no PR/patch created.
- `skipped_no_acceptable_proposal`: Attempts exhausted without passing the gate.
- `blocked_no_usable_sources`: No sources were usable (blocked/offline/error).
- `completed_no_pr`: Accepted changes with patch output (no GitHub token/repo).
- `pr_opened`: Accepted changes with PR created.

## Safety

- PromptOps is disabled by default.
- Only curate trusted sources.
- Evals enforce a multi-metric gate before PR creation.
