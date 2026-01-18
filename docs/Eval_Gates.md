# Eval Gates

Phase 0 introduces a deterministic retrieval evaluation gate. It runs quickly and uses a synthetic, privacy-safe fixture corpus.

## Deterministic Mode
Deterministic mode:
- Uses a small synthetic corpus seeded at runtime.
- Reads a JSONL dataset of evaluation cases.
- Computes recall@k, MRR, and no-evidence accuracy.
- Fails the gate if metrics regress below the committed baseline thresholds.

## Dataset Format (JSONL)
Each line is a JSON object with:
- id
- scenario_class
- query
- expected_evidence_ids (list) or expected_no_evidence (boolean)
- time_window (optional; start and end ISO timestamps)

Example line:
`{"id":"case-1","scenario_class":"lexical","query":"roadmap","expected_evidence_ids":["EVT-1"]}`

## Extended Mode
Extended mode is optional and local-only. If enabled, it runs the same harness with a larger or richer dataset. If extended mode is not configured, the harness falls back to deterministic mode and logs a notice.

## Updating Baselines
To update thresholds:
- Run the deterministic eval locally.
- Compare metrics to the current baseline values.
- Update the baseline file when improvements are verified.
