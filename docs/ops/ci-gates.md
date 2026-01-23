# CI Gates (Next-10)

This repo enforces deterministic gates for pillars.

## Gates
- Pillar declaration: `tools/pillar_gate.py`
- Privacy regression scanner: `tools/privacy_scanner.py`
- Provenance chain verifier: `tools/provenance_gate.py`
- Coverage regression: `tools/coverage_gate.py`
- Latency budget regression: `tools/latency_gate.py`
- Retrieval sensitivity ±1: `tools/retrieval_sensitivity.py`
- No-evidence determinism: `tools/no_evidence_gate.py`
- Conflict scenario suite: `tools/conflict_gate.py`
- Citation integrity simulation: `tools/integrity_gate.py`

## Output artifacts
- `artifacts/privacy_report.json`
- `artifacts/coverage_report.json`
- `artifacts/latency_report.json`
- `artifacts/instability_report.json`
- `artifacts/provenance_report.json`
- `artifacts/no_evidence_report.json`
- `artifacts/conflict_report.json`
- `artifacts/integrity_report.json`

## Required testing
Always run this before moving on to the next task or running the gates:
```bash
poetry install --with dev
poetry run pytest -q
```

## Local run
```bash
poetry run python tools/pillar_gate.py
poetry run python tools/privacy_scanner.py
poetry run python tools/provenance_gate.py
poetry run python tools/coverage_gate.py
poetry run python tools/latency_gate.py
poetry run python tools/retrieval_sensitivity.py
poetry run python tools/no_evidence_gate.py
poetry run python tools/conflict_gate.py
poetry run python tools/integrity_gate.py
```
