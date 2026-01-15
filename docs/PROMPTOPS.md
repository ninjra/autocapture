# PromptOps

PromptOps automates prompt refreshes by ingesting sources, proposing prompt updates,
running evals, and optionally opening GitHub pull requests. Phase 2 adds a multi-attempt,
agentic loop with a strict, deterministic acceptance gate that requires at least one
metric improvement.

## How it works

1. **Fetch sources**
   - Configure sources in `promptops.sources` (HTTP(S) URLs or local file paths).
   - Snapshots are stored under `<data_dir>/promptops/sources/<run_id>/`.
   - Offline mode blocks network sources unless `privacy.cloud_enabled=true`.
2. **Propose prompt updates (multi-attempt)**
   - PromptOps reads prompt YAML files from `prompts/raw/`.
   - It calls the configured LLM provider (`routing.llm`) to generate proposed prompt YAML.
   - Each attempt can incorporate feedback from the previous attempt (baseline metrics,
     gate results, and compact failure summaries) to steer refinements.
   - Versions are incremented only when there is a semantic change to prompt content.
   - Derived prompts are compiled into `autocapture/prompts/derived/`.
3. **Run evals + gates**
   - Uses `evals/golden_queries.json`.
   - Runs multiple baseline/proposed evals (`promptops.eval_repeats`) and aggregates
     via `promptops.eval_aggregation` (default: `worst_case`).
   - Applies a multi-metric gate:
     - **Absolute thresholds** (floors/ceilings) for each metric.
     - **No-regression tolerances** vs. baseline on every gated metric.
     - **Required improvement**: at least one metric must improve by a minimum delta.
   - Among passing candidates, PromptOps deterministically selects the best candidate
     (lexicographic ordering: higher verifier pass rate, higher citation coverage,
     lower refusal rate, lower latency).
4. **Create a PR or patch artifact**
   - If `promptops.github_token` and `promptops.github_repo` are set, PromptOps
     creates a branch and opens a PR.
   - Otherwise, a unified diff is written to `<data_dir>/promptops/patches/`.
   - If no candidate passes the gate, PromptOps records a successful noop/skip.

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
  max_attempts: 3
  early_stop_on_first_accept: false
  acceptance_tolerance: 0.02
  eval_repeats: 3
  eval_aggregation: "worst_case"
  require_improvement: true
  min_improve_verifier_pass_rate: 0.01
  min_improve_citation_coverage: 0.01
  min_improve_refusal_rate: 0.01
  min_improve_mean_latency_ms: 100.0
  tolerance_citation_coverage: 0.02
  tolerance_refusal_rate: 0.02
  tolerance_mean_latency_ms: 250.0
  min_verifier_pass_rate: 0.60
  min_citation_coverage: 0.60
  max_refusal_rate: 0.30
  max_mean_latency_ms: 15000.0
  pr_cooldown_hours: 0.0
  max_source_bytes: 1048576
  max_source_excerpt_chars: 2000
  max_sources: 32
```

## Statuses

PromptOps records the run outcome as a status:

- `noop_no_semantic_change`: Attempt produced no prompt diffs beyond whitespace.
- `skipped_no_acceptable_proposal`: Attempts exhausted without passing the gate.
- `blocked_no_usable_sources`: No sources were usable (blocked/offline/error).
- `completed_no_pr`: Accepted changes with patch output (no GitHub token/repo).
- `pr_opened`: Accepted changes with PR created.
PromptOps treats noop/skip outcomes as successful runs (exit code 0).

## Guardrails & safety

- **Prompt injection resistance**: sources are treated as untrusted input, and instructions
  in sources are explicitly ignored.
- **Strict YAML validation**: required fields, types, placeholder preservation, and size
  limits are enforced before evaluation.
- **No churn acceptance**: candidates must pass thresholds, avoid regressions, and improve
  at least one metric by a minimum delta.
- **No PR on failure**: PromptOps never opens a PR unless a candidate passes the gate.
- **Optional PR cooldown**: prevent spam by setting `promptops.pr_cooldown_hours`.
- **Deterministic selection**: when multiple candidates pass, PromptOps picks the best
  candidate with a deterministic tie-breaker to keep runs reproducible.

### Defaults

- PromptOps is disabled by default.
- Only curate trusted sources.
- Evals enforce a multi-metric gate before PR creation.
