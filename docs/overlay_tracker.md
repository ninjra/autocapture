# Hotness Overlay Tracker (Windows)

## Overview
The Hotness Overlay Tracker is a Windows-only module that records **metadata-only** activity
signals (foreground window changes and input activity) into an append-only local event log.
It surfaces a right-docked overlay UI that ranks items by hotness (exponential decay) and
buckets them into **Active (<= 48h)** vs **Stale (> 48h)**.

**Not a ChatGPT UI scraper.** The overlay tracker does not access or enumerate ChatGPT UI
conversation history. OpenAI conversation state is managed on the server side and is not
readable from local UI state. See the guide here:
https://platform.openai.com/docs/guides/conversation-state

## Security posture (metadata-only)
- **No keystrokes or keycodes.** Input is detected via GetLastInputInfo (activity only),
  never via low-level keyboard hooks.
- **No clipboard, DOM, or tab enumeration.** The module only tracks timestamps,
  process name, and window title; optional URL tracking is disabled by default
  and fails closed to title identity.
- **No admin privileges required.** All collectors run in user space.

Windows API references used by the collectors:
- SetWinEventHook (EVENT_SYSTEM_FOREGROUND):
  https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwineventhook
- GetLastInputInfo:
  https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getlastinputinfo
- LASTINPUTINFO:
  https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-lastinputinfo
- RegisterHotKey:
  https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-registerhotkey
- WM_HOTKEY:
  https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-hotkey

## Enablement
Set the following in your config:

```yaml
overlay_tracker:
  enabled: true
  platforms: ["windows"]
  ui:
    enabled: true
```

The module is fully disable-able: if `overlay_tracker.enabled=false`, no collectors
are started, no UI is shown, and no DB writes are made.

Note: the overlay UI requires `autocapture app` so the Qt event loop is running.

CLI status:
`autocapture overlay-tracker status`

## Data model (append-only evidence log)
Tables created:
- `overlay_projects`: project containers (default: Inbox)
- `overlay_items`: per-identity current state
- `overlay_item_identities`: identity key -> item mapping
- `overlay_events`: **append-only** event log (evidence source of truth)
- `overlay_kv`: retention/health metadata

The overlay UI derives hotness and stale/active buckets from `overlay_events` +
`overlay_items.last_activity_at_utc`.

## Retention
Configured in `overlay_tracker.retention`:
- `event_days`: delete events older than N days (default 14)
- `event_cap`: hard cap on event count (default 200,000)

Retention deletes events only; items remain until they age out naturally.

## Hotkeys & interactive mode
Hotkeys are configurable in `overlay_tracker.hotkeys`.
If a hotkey cannot be registered, deterministic fallbacks are attempted
(e.g., add Shift/Alt or use F24). See logs for the final mapping.

The overlay is **click-through by default**. Use the interactive hotkey to enable
mouse interaction for 10 seconds.

## Troubleshooting
- **Overlay not visible:** ensure the app is started via `autocapture app` so the Qt
  event loop is running.
- **Hotkey conflicts:** customize hotkeys in config; the manager logs fallbacks.
- **Overlay hides in fullscreen:** disable `overlay_tracker.ui.auto_hide_fullscreen`.
- **Database locked warnings:** overlay tracker uses retry/backoff and batching; ensure
  the DB is on a local disk.
