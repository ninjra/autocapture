# Checksum Policy

Native extensions must be allowlisted in `CHECKSUMS.json` with their SHA-256 checksums.
`tools/verify_checksums.py` enforces the allowlist in CI, and runtime loads only verified
extensions. When `security.secure_mode=true`, checksum mismatches or unknown extensions
fail closed.

## Workflow
- Add the extension binary to the repo or distribution.
- Compute the SHA-256 checksum.
- Update `CHECKSUMS.json` with the path and checksum.
- Run `python tools/verify_checksums.py` to confirm.

## Notes
- Do not add or resolve `[MISSING_VALUE]` placeholders unless explicitly provided.
