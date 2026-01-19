# Citations & Provenance

## Ledger chain
- Each answer has a hash-chained ledger stream.
- `entry_hash = SHA256(canonical_json(entry) + prev_hash)`.

## Required chain
capture → extract → index → retrieve → answer

## Emission rules
- A citation may only be emitted if the provenance chain for the span exists for the answer.
- If the chain is missing, the span is marked non-citable and the answer is replanned.

## Integrity checks
- Span exists and is not tombstoned.
- Frame exists and media is readable (and decryptable if encrypted).
- Optional sampled hash validation.

