# PromptOps

PromptOps keeps prompt templates up to date by ingesting curated sources and proposing changes.

## How it works
- Sources are configured in `promptops.sources`.
- PromptOps fetches sources, stores raw items, and generates proposals.
- Proposals update `/prompts/raw/*.yaml` and compiled `/prompts/derived/*.yaml`.
- Evaluations run against the local eval suite.

## PR Workflow
- When configured with `promptops.github_token` and `promptops.github_repo`, PromptOps opens a PR.
- The PR includes:
  - Prompt changes
  - Eval report
  - Rollback notes

## Safety
- PromptOps is disabled by default.
- Only curated sources should be enabled.
