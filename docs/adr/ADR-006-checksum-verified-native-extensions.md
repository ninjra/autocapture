# ADR-006: Checksum-Verified Native Extensions
Date: 2026-01-21
Status: Accepted

## Context
Native extension loading must be verifiable and fail closed in secure mode.

## Decision
- Introduce `CHECKSUMS.json` and `CHECKSUM_POLICY.md` as the allowlist source.
- Provide `tools/verify_checksums.py` for CI verification.
- Wrap native extension loading with checksum verification and enforce `security.secure_mode`.

## Consequences
- Native extensions only load when their checksum is allowlisted.
- Secure mode blocks unknown or mismatched extensions.
