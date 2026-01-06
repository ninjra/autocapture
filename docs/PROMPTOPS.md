# PromptOps

PromptOps ingests curated sources and stores raw snapshots for later review.

## How it works
- Sources are configured in `promptops.sources`.
- PromptOps fetches sources and stores raw items for auditing.
- It does not yet mutate prompts or run automated evals.

## PR Workflow
- PromptOps does not open PRs in this build.

## Safety
- PromptOps is disabled by default.
- Only curated sources should be enabled.
