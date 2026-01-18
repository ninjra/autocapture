## Summary (M1–M8)
- [ ] M1: Foreground + input collectors (Windows user-space)
- [ ] M2: Right-docked overlay UI (click-through + interactive mode)
- [ ] M3: Deterministic identity + hotness decay + 48h stale bucketing
- [ ] M4: Append-only event log + evidence/audit view
- [ ] M5: Metadata-only collection (no keylogging/clipboard/DOM)
- [ ] M6: Enablement flags + platform gating
- [ ] M7: Retention policy (days + cap)
- [ ] M8: CI / regression tests

## Enablement & defaults
- `overlay_tracker.enabled`:
- `overlay_tracker.ui.enabled`:
- `overlay_tracker.url_plugin.enabled`:

## Retention
- `overlay_tracker.retention.event_days`:
- `overlay_tracker.retention.event_cap`:

## Evidence/audit UX
- Describe how evidence is sourced from append-only `overlay_events`:

## Pillar enforcement checklist
**Improved:**
- [ ] P1 performant (bounded polling, debounced writes, retention cap)
- [ ] P2 accurate (deterministic identity/decay + 48h boundary tests)
- [ ] P3 secure (metadata-only, deny/allow lists, no keylogging)
- [ ] P4 citable (append-only log + evidence view)

**Risked:**
- [ ] P1 UI thread blocking / event volume
- [ ] P2 title normalization churn
- [ ] P3 over-collection
- [ ] P4 evidence mismatch

**Enforcement locations:**
- Core:
- Collectors:
- Policy:
- UI:
- Cleaner:

**Regression tests / CI gates:**
- Unit tests:
- Integration tests:
- Security tests:
- Windows job:

## References (plain URLs)
- https://platform.openai.com/docs/guides/conversation-state
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwineventhook
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getlastinputinfo
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-registerhotkey
- https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-lastinputinfo
- https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-hotkey
