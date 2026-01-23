"""Settings schema builder for UX."""

from __future__ import annotations

import datetime as dt

from .models import SettingsField, SettingsOption, SettingsSchema, SettingsSection, SettingsTier


def build_settings_schema() -> SettingsSchema:
    tiers = [
        SettingsTier(
            tier_id="guided",
            label="Guided",
            description="Common day-to-day settings with safe defaults.",
            rank=1,
        ),
        SettingsTier(
            tier_id="advanced",
            label="Advanced",
            description="Advanced routing and privacy controls.",
            rank=2,
        ),
        SettingsTier(
            tier_id="expert",
            label="Expert",
            description="Potentially risky changes. Requires explicit confirmation.",
            rank=3,
            requires_confirm=True,
            confirm_phrase="I UNDERSTAND THIS MAY AFFECT PRIVACY",
        ),
    ]

    sections = [
        SettingsSection(
            section_id="presets",
            label="Presets",
            description="Quickly switch between bundled profiles.",
            fields=[
                SettingsField(
                    path="active_preset",
                    label="Active preset",
                    description="Choose the default capture + privacy preset.",
                    kind="select",
                    tier="guided",
                    options=[
                        SettingsOption(label="Privacy first", value="privacy_first"),
                        SettingsOption(label="High fidelity", value="high_fidelity"),
                    ],
                )
            ],
        ),
        SettingsSection(
            section_id="routing",
            label="Routing",
            description="Select the default providers for each stage.",
            fields=[
                SettingsField(
                    path="routing.llm",
                    label="LLM provider",
                    description="Provider used for final answers.",
                    kind="select",
                    tier="guided",
                    options_source="plugins:llm.provider",
                ),
                SettingsField(
                    path="routing.ocr",
                    label="OCR engine",
                    description="OCR engine for new captures.",
                    kind="select",
                    tier="guided",
                    options_source="plugins:ocr.engine",
                ),
                SettingsField(
                    path="routing.embedding",
                    label="Embedding model",
                    description="Embedding provider for indexing.",
                    kind="select",
                    tier="guided",
                    options_source="plugins:embedder.text",
                ),
                SettingsField(
                    path="routing.retrieval",
                    label="Retrieval strategy",
                    description="Primary retrieval strategy.",
                    kind="select",
                    tier="guided",
                    options_source="plugins:retrieval.strategy",
                ),
                SettingsField(
                    path="routing.reranker",
                    label="Reranker",
                    description="Optional reranker stage.",
                    kind="select",
                    tier="advanced",
                    options_source="plugins:reranker",
                ),
                SettingsField(
                    path="routing.compressor",
                    label="Compressor",
                    description="Context compression strategy.",
                    kind="select",
                    tier="advanced",
                    options_source="plugins:compressor",
                ),
                SettingsField(
                    path="routing.verifier",
                    label="Verifier",
                    description="Verification stage.",
                    kind="select",
                    tier="advanced",
                    options_source="plugins:verifier",
                ),
                SettingsField(
                    path="routing.capture",
                    label="Capture pipeline",
                    description="Capture routing (if extensions available).",
                    kind="select",
                    tier="advanced",
                    options_source="plugins:capture.pipeline",
                ),
            ],
        ),
        SettingsSection(
            section_id="privacy",
            label="Privacy",
            description="Capture pause, redaction defaults, and cloud controls.",
            fields=[
                SettingsField(
                    path="privacy.paused",
                    label="Capture paused",
                    description="Pause capture immediately.",
                    kind="bool",
                    tier="guided",
                ),
                SettingsField(
                    path="privacy.snooze_until_utc",
                    label="Snooze until (UTC)",
                    description="Pause capture until the provided ISO timestamp.",
                    kind="string",
                    tier="guided",
                    placeholder="2026-01-20T18:30:00Z",
                ),
                SettingsField(
                    path="privacy.sanitize_default",
                    label="Sanitize by default",
                    description="Redact sensitive text in answers by default.",
                    kind="bool",
                    tier="guided",
                ),
                SettingsField(
                    path="privacy.extractive_only_default",
                    label="Extractive-only by default",
                    description="Prefer extractive answers unless explicitly disabled.",
                    kind="bool",
                    tier="guided",
                ),
                SettingsField(
                    path="privacy.exclude_processes",
                    label="Excluded processes",
                    description="Process names to skip capture.",
                    kind="list",
                    tier="advanced",
                    placeholder="one process per line",
                ),
                SettingsField(
                    path="privacy.exclude_window_title_regex",
                    label="Excluded window title regex",
                    description="Regex patterns for window titles to skip.",
                    kind="list",
                    tier="advanced",
                    placeholder="one regex per line",
                ),
                SettingsField(
                    path="privacy.exclude_monitors",
                    label="Excluded monitors",
                    description="Monitor identifiers to skip capture.",
                    kind="list",
                    tier="advanced",
                    placeholder="one monitor id per line",
                ),
                SettingsField(
                    path="privacy.exclude_regions",
                    label="Excluded regions (JSON)",
                    description="List of region objects to skip capture.",
                    kind="json",
                    tier="advanced",
                    multiline=True,
                ),
                SettingsField(
                    path="privacy.mask_regions",
                    label="Mask regions (JSON)",
                    description="List of region objects to mask.",
                    kind="json",
                    tier="advanced",
                    multiline=True,
                ),
                SettingsField(
                    path="privacy.cloud_enabled",
                    label="Allow cloud providers",
                    description="Allow cloud processing where supported.",
                    kind="bool",
                    tier="expert",
                    danger_level="danger",
                ),
                SettingsField(
                    path="privacy.allow_cloud_images",
                    label="Allow cloud images",
                    description="Allow sending images to cloud vision providers.",
                    kind="bool",
                    tier="expert",
                    danger_level="danger",
                ),
                SettingsField(
                    path="privacy.allow_token_vault_decrypt",
                    label="Allow token vault decrypt",
                    description="Permit API decrypt of stored tokens.",
                    kind="bool",
                    tier="expert",
                    danger_level="danger",
                ),
            ],
        ),
        SettingsSection(
            section_id="llm",
            label="Answer strategy",
            description="Control prompt strategy and reasoning defaults.",
            fields=[
                SettingsField(
                    path="llm.prompt_strategy_default",
                    label="Prompt strategy",
                    description="Default prompt strategy for answers.",
                    kind="select",
                    tier="advanced",
                    options=[
                        SettingsOption(label="Baseline", value="baseline"),
                        SettingsOption(label="Repeat 2x", value="repeat_2x"),
                        SettingsOption(label="Repeat 3x", value="repeat_3x"),
                    ],
                ),
                SettingsField(
                    path="llm.strategy_auto_mode",
                    label="Auto strategy",
                    description="Let the system adjust prompt strategy automatically.",
                    kind="bool",
                    tier="advanced",
                ),
                SettingsField(
                    path="llm.enable_step_by_step",
                    label="Step-by-step reasoning",
                    description="Enable step-by-step reasoning mode.",
                    kind="bool",
                    tier="advanced",
                ),
                SettingsField(
                    path="llm.step_by_step_phrase",
                    label="Step-by-step phrase",
                    description="Phrase appended when step-by-step is enabled.",
                    kind="string",
                    tier="advanced",
                ),
                SettingsField(
                    path="llm.prompt_store_redaction",
                    label="Redact stored prompts",
                    description="Redact stored prompts when persisted to disk.",
                    kind="bool",
                    tier="expert",
                ),
            ],
        ),
        SettingsSection(
            section_id="tracking",
            label="Tracking",
            description="Host event tracking and activity sensors.",
            fields=[
                SettingsField(
                    path="tracking.enabled",
                    label="Tracking enabled",
                    description="Enable host event tracking.",
                    kind="bool",
                    tier="advanced",
                ),
                SettingsField(
                    path="tracking.track_mouse_movement",
                    label="Track mouse movement",
                    description="Capture mouse movement events.",
                    kind="bool",
                    tier="advanced",
                ),
                SettingsField(
                    path="tracking.raw_event_stream_enabled",
                    label="Raw input event stream",
                    description="Persist raw keyboard/mouse events for action replay timelines.",
                    kind="bool",
                    tier="expert",
                ),
                SettingsField(
                    path="tracking.enable_clipboard",
                    label="Track clipboard",
                    description="Capture clipboard events.",
                    kind="bool",
                    tier="expert",
                ),
                SettingsField(
                    path="tracking.retention_days",
                    label="Tracking retention days",
                    description="Days to retain host event tracking data.",
                    kind="int",
                    tier="expert",
                    placeholder="Leave blank for default",
                ),
                SettingsField(
                    path="tracking.raw_event_retention_days",
                    label="Raw input retention days",
                    description="Days to retain raw input events.",
                    kind="int",
                    tier="expert",
                    placeholder="Leave blank for default",
                ),
            ],
        ),
    ]

    return SettingsSchema(
        schema_version=1,
        generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        tiers=tiers,
        sections=sections,
    )
