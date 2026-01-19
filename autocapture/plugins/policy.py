"""Policy gate enforcement for plugin extensions."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from ..config import AppConfig, ModelStageConfig, is_loopback_host
from .errors import PluginPolicyError
from .manifest import ExtensionManifestV1


@dataclass(frozen=True)
class PolicyDecision:
    ok: bool
    reason: str | None = None


class PolicyGate:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def guard_cloud_text(
        self,
        *,
        stage: str | None,
        stage_config: ModelStageConfig | None,
        provider: str,
        base_url: str | None,
        cloud: bool,
    ) -> None:
        if not cloud:
            return
        if stage_config is not None and not stage_config.allow_cloud:
            raise PluginPolicyError(
                f"Cloud provider blocked for stage '{stage}'. "
                f"Set model_stages.{stage}.allow_cloud=true to allow."
            )
        if self._config.offline:
            raise PluginPolicyError("Cloud provider blocked because offline=true.")
        if not self._config.privacy.cloud_enabled:
            raise PluginPolicyError("Cloud provider blocked because privacy.cloud_enabled=false.")

    def guard_cloud_images(
        self,
        *,
        provider: str,
        base_url: str | None,
        allow_cloud: bool,
    ) -> None:
        if _is_local_endpoint(base_url) and provider != "openai":
            return
        if not allow_cloud:
            raise PluginPolicyError("Cloud vision calls not permitted by backend config")
        if self._config.offline:
            raise PluginPolicyError("Cloud vision calls not permitted because offline=true")
        if not self._config.privacy.cloud_enabled:
            raise PluginPolicyError(
                "Cloud vision calls not permitted because privacy.cloud_enabled=false"
            )
        if not self._config.privacy.allow_cloud_images:
            raise PluginPolicyError(
                "Cloud vision calls not permitted because allow_cloud_images=false"
            )

    def enforce_extension_policy(
        self,
        *,
        stage: str | None,
        stage_config: ModelStageConfig | None,
        extension: ExtensionManifestV1,
    ) -> PolicyDecision:
        pillars = extension.pillars
        if not pillars or not pillars.data_handling:
            return PolicyDecision(ok=True)
        handling = pillars.data_handling
        requires_cloud = handling.cloud == "required" or (
            handling.cloud == "optional" and not handling.supports_local
        )
        requires_cloud_images = handling.cloud_images == "required" or (
            handling.cloud_images == "optional" and not handling.supports_local
        )
        if requires_cloud:
            try:
                self.guard_cloud_text(
                    stage=stage,
                    stage_config=stage_config,
                    provider=extension.id,
                    base_url=None,
                    cloud=True,
                )
            except PluginPolicyError as exc:
                return PolicyDecision(ok=False, reason=str(exc))
        if requires_cloud_images:
            try:
                self.guard_cloud_images(
                    provider=extension.id,
                    base_url=None,
                    allow_cloud=True,
                )
            except PluginPolicyError as exc:
                return PolicyDecision(ok=False, reason=str(exc))
        return PolicyDecision(ok=True)


def _is_local_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return True
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return is_loopback_host(host)
