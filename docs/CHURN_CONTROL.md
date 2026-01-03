# Churn Control: Frozen Surfaces

Frozen Surfaces lock down completed subsystems by recording their exact content
hashes in a manifest. When a file is frozen, any change to its contents fails
unit tests until the file is explicitly unfrozen. This prevents accidental or
casual refactors that reintroduce churn.

## How it works

- The manifest lives at `autocapture/stability/frozen_manifest.json`.
- Each frozen file is recorded with a SHA256 hash, timestamp, and reason.
- Tests compare current file content to the manifest and fail on mismatches.

## Commands

Use the CLI helper:

```bash
python tools/freeze_surfaces.py freeze --reason "<reason>" path/to/file.py
python tools/freeze_surfaces.py unfreeze --reason "<reason>" path/to/file.py
python tools/freeze_surfaces.py verify
```

## Rules

- Never refactor frozen files for cleanliness.
- Only change frozen files to fix correctness/security issues.
- Any change to a frozen file requires an explicit unfreeze with a reason.
- Prefer working in unfrozen areas. Freeze only when stable and tested.

This policy keeps subsystems stable and avoids refactor churn between loops.
