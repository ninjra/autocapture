# 1. System Context & Constraints

## Purpose
Provide a deterministic, self-contained reference for the Autocapture system's scope, constraints, and
validation rules. This file is the single source of truth for machine-reasoning alignment.

## Scope
- Product: Autocapture (local-first activity capture + recall).
- Repo: autocapture (Python 3.12, Poetry).
- Runtime target: Windows 11 host with WSL2 Linux guest.
- Offline-capable: do not assume network access.

## Non-Goals
- No external links or citations.
- No speculative future roadmap beyond explicit placeholders.

## Operating Context
- Host OS: Windows 11.
- Guest OS: WSL2 Linux.
- Execution: CLI + background worker + local API + tray UI.
- Default run mode: local storage and local inference, with optional external services disabled by default.

## Hard Constraints
- Deterministic wording and structure.
- No external URLs (http/https) anywhere in this file.
- Preserve [MISSING_VALUE] markers verbatim.
- Keep changes minimal and aligned to the request.
- Do not embed secrets or machine-specific paths.

## Enforcement_Location
- Documentation_Artifact: BLUEPRINT.md
- Validator: [MISSING_VALUE]
- Test_Hook: [MISSING_VALUE]

## Validation_Checklist
- [ ] Section ordering matches required headings (#1..#4).
- [ ] No external links are present.
- [ ] Source_Index and Coverage_Map are internally consistent.
- [ ] All [MISSING_VALUE] markers remain until resolved.
- [ ] No secrets or machine-specific paths included.

## Source_Index:
- SRC-001: README.md
- SRC-002: pyproject.toml
- SRC-003: autocapture/main.py
- SRC-004: autocapture/config.py
- SRC-005: docs/pillars/
- SRC-006: tests/

## Coverage_Map:
- 1.System_Context_And_Constraints -> SRC-001, SRC-002
- 2.Functional_Modules_And_Logic -> SRC-003, SRC-004
- 3.ADRs -> SRC-005
- 4.Grounding_Data -> SRC-006

# 2. Functional Modules & Logic

## Module Inventory
- Capture Pipeline: [MISSING_VALUE]
- OCR + Vision: [MISSING_VALUE]
- Storage + Retention: [MISSING_VALUE]
- Retrieval + Ranking: [MISSING_VALUE]
- LLM Routing + Answering: [MISSING_VALUE]
- API + UI: [MISSING_VALUE]
- Observability + Health: [MISSING_VALUE]

## Data Flow (High Level)
1) Capture -> [MISSING_VALUE]
2) OCR/Enrichment -> [MISSING_VALUE]
3) Storage -> [MISSING_VALUE]
4) Retrieval -> [MISSING_VALUE]
5) Answering -> [MISSING_VALUE]
6) API/UI -> [MISSING_VALUE]

## Guardrails
- Privacy defaults: [MISSING_VALUE]
- Offline behavior: [MISSING_VALUE]
- Failure modes: [MISSING_VALUE]

# 3. Architecture Decision Records (ADRs)

## ADR_Index
- ADR-001: [MISSING_VALUE] (Status: Proposed) (Date: [MISSING_VALUE])
- ADR-002: [MISSING_VALUE] (Status: Proposed) (Date: [MISSING_VALUE])
- ADR-003: [MISSING_VALUE] (Status: Proposed) (Date: [MISSING_VALUE])

## ADR_Details

### ADR-001
- Title: [MISSING_VALUE]
- Context: [MISSING_VALUE]
- Decision: [MISSING_VALUE]
- Consequences: [MISSING_VALUE]

### ADR-002
- Title: [MISSING_VALUE]
- Context: [MISSING_VALUE]
- Decision: [MISSING_VALUE]
- Consequences: [MISSING_VALUE]

### ADR-003
- Title: [MISSING_VALUE]
- Context: [MISSING_VALUE]
- Decision: [MISSING_VALUE]
- Consequences: [MISSING_VALUE]

# 4. Grounding Data (Few-Shot Samples)

## Format
Each sample must include:
- Input
- Expected_Output
- Evidence (SRC-### only; no external links)

## Sample_001
- Input: [MISSING_VALUE]
- Expected_Output: [MISSING_VALUE]
- Evidence: SRC-001

## Sample_002
- Input: [MISSING_VALUE]
- Expected_Output: [MISSING_VALUE]
- Evidence: SRC-003

## Sample_003
- Input: [MISSING_VALUE]
- Expected_Output: [MISSING_VALUE]
- Evidence: SRC-004
