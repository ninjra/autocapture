# Privacy Invariants

## Hard invariants (SPEC-260117)
- Excluded frames are never persisted as pixels and never create artifacts/spans/index entries.
- Masked frames are irreversibly obscured before hashing/encryption/persistence.
- Offline-first: no background upload; no cloud-default indexing.

## Implementation summary
- Policy gate evaluates process/title/monitor and screen lock status before enqueue.
- Excluded frames are recorded with `media_path = NULL`, `privacy_flags.excluded = true` and do not enter OCR or indexing.
- Masked regions (`privacy.exclude_regions` / `privacy.mask_regions`) are applied before frame hashing and media encryption.
- Media at rest is encrypted with AES-GCM; SQLCipher is preferred for SQLite metadata.

## Secure-mode gate
- `database.secure_mode_required = true` (default) refuses to start if SQLite is unencrypted.
- Development opt-out requires explicit `database.allow_insecure_dev = true`.

## Local scanner
Run the privacy regression scanner locally:

```bash
poetry run python tools/privacy_scanner.py
```

Reports:
- `artifacts/privacy_report.json`
