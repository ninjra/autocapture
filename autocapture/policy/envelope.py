"""Policy enforcement envelope for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar
from urllib.parse import urlparse
import inspect

from ..config import AppConfig, is_loopback_host
from ..logging_utils import get_logger
from ..llm.providers import LLMProvider
from ..memory.prompt_injection import scan_prompt_injection

T = TypeVar("T")


@dataclass(frozen=True)
class PolicyDecision:
    cloud: bool
    allow_cloud: bool | None


class PolicyEnvelope:
    def __init__(self, config: AppConfig | None) -> None:
        self._config = config
        self._log = get_logger("policy.envelope")

    async def execute_stage(
        self,
        *,
        stage: str | None,
        provider: LLMProvider,
        decision: object | None,
        system_prompt: str,
        user_prompt: str,
        context_pack_text: str,
        temperature: float | None = None,
        priority: str = "foreground",
        warnings: list[str] | None = None,
        evidence: list[object] | None = None,
    ) -> str:
        policy = self._evaluate_text_policy(stage=stage, provider=provider, decision=decision)
        self._enforce_cloud_policy(stage, policy)
        self._enforce_injection_policy(evidence, warnings)
        try:
            return await provider.generate_answer(
                system_prompt,
                user_prompt,
                context_pack_text,
                temperature=temperature,
                priority=priority,
            )
        except TypeError as exc:
            if "priority" not in str(exc):
                raise
        return await provider.generate_answer(
            system_prompt,
            user_prompt,
            context_pack_text,
            temperature=temperature,
        )

    async def execute_call(
        self,
        *,
        stage: str | None,
        call: Callable[[], Awaitable[T] | T],
        cloud: bool,
        allow_cloud: bool | None = None,
        warnings: list[str] | None = None,
        evidence: list[object] | None = None,
    ) -> T:
        policy = PolicyDecision(cloud=cloud, allow_cloud=allow_cloud)
        self._enforce_cloud_policy(stage, policy)
        self._enforce_injection_policy(evidence, warnings)
        result = call()
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result  # type: ignore[return-value]

    def execute_call_sync(
        self,
        *,
        stage: str | None,
        call: Callable[[], T],
        cloud: bool,
        allow_cloud: bool | None = None,
        warnings: list[str] | None = None,
        evidence: list[object] | None = None,
    ) -> T:
        policy = PolicyDecision(cloud=cloud, allow_cloud=allow_cloud)
        self._enforce_cloud_policy(stage, policy)
        self._enforce_injection_policy(evidence, warnings)
        return call()

    def execute_vision_sync(
        self,
        *,
        stage: str | None,
        call: Callable[[], T],
        cloud: bool,
        allow_cloud: bool | None = None,
        warnings: list[str] | None = None,
    ) -> T:
        policy = PolicyDecision(cloud=cloud, allow_cloud=allow_cloud)
        self._enforce_cloud_policy(stage, policy)
        self._enforce_cloud_images_policy(stage, policy)
        self._enforce_injection_policy(None, warnings)
        return call()

    def infer_cloud_from_endpoint(self, base_url: str | None, provider: str | None) -> bool:
        if self._config is not None and getattr(self._config.security, "provider", "") == "test":
            return False
        if provider == "openai":
            return True
        if not base_url:
            return False
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        if not host:
            return False
        return not is_loopback_host(host)

    def _evaluate_text_policy(
        self, *, stage: str | None, provider: LLMProvider, decision: object | None
    ) -> PolicyDecision:
        cloud = _infer_cloud(provider, decision)
        allow_cloud = self._stage_allows_cloud(stage)
        return PolicyDecision(cloud=cloud, allow_cloud=allow_cloud)

    def _stage_allows_cloud(self, stage: str | None) -> bool | None:
        if not stage or self._config is None:
            return None
        stages = self._config.model_stages
        mapping = {
            "query_refine": stages.query_refine,
            "draft_generate": stages.draft_generate,
            "final_answer": stages.final_answer,
            "tool_transform": stages.tool_transform,
            "entailment_judge": stages.entailment_judge,
        }
        stage_cfg = mapping.get(stage)
        if stage_cfg is None:
            return None
        return bool(stage_cfg.allow_cloud)

    def _enforce_cloud_policy(self, stage: str | None, policy: PolicyDecision) -> None:
        if not policy.cloud:
            return
        if policy.allow_cloud is False:
            if stage:
                raise RuntimeError(
                    f"Cloud provider blocked for stage '{stage}'. "
                    f"Set model_stages.{stage}.allow_cloud=true to allow."
                )
            raise RuntimeError("Cloud provider blocked (allow_cloud=false).")
        if self._config is None:
            return
        if self._config.offline:
            raise RuntimeError("Cloud provider blocked because offline=true.")
        if not self._config.privacy.cloud_enabled:
            raise RuntimeError("Cloud provider blocked because privacy.cloud_enabled=false.")

    def _enforce_injection_policy(
        self, evidence: list[object] | None, warnings: list[str] | None
    ) -> None:
        if self._config is None:
            return
        policy = getattr(self._config, "policy", None)
        if policy is None or not getattr(policy, "enforce_prompt_injection", False):
            return
        if not evidence:
            return
        max_risk = 0.0
        for item in evidence:
            risk = 0.0
            try:
                risk = float(getattr(item, "injection_risk", 0.0) or 0.0)
                text = getattr(item, "text", None)
                if text:
                    scan = scan_prompt_injection(str(text))
                    risk = max(risk, float(scan.risk_score))
            except Exception:
                continue
            max_risk = max(max_risk, risk)
        if max_risk >= float(getattr(policy, "prompt_injection_block_threshold", 1.0)):
            if warnings is not None:
                warnings.append("prompt_injection_blocked")
            raise RuntimeError("Prompt injection risk too high; blocking LLM call")
        if max_risk >= float(getattr(policy, "prompt_injection_warn_threshold", 1.0)):
            if warnings is not None:
                warnings.append("prompt_injection_warning")
            self._log.warning("Prompt injection risk elevated (risk={})", max_risk)

    def _enforce_cloud_images_policy(self, stage: str | None, policy: PolicyDecision) -> None:
        if not policy.cloud:
            return
        if policy.allow_cloud is False:
            if stage:
                raise RuntimeError(
                    f"Cloud vision blocked for stage '{stage}'. "
                    f"Set model_stages.{stage}.allow_cloud=true to allow."
                )
            raise RuntimeError("Cloud vision blocked (allow_cloud=false).")
        if self._config is None:
            return
        if self._config.offline:
            raise RuntimeError("Cloud vision calls not permitted because offline=true")
        if not self._config.privacy.cloud_enabled:
            raise RuntimeError(
                "Cloud vision calls not permitted because privacy.cloud_enabled=false"
            )
        if not self._config.privacy.allow_cloud_images:
            raise RuntimeError(
                "Cloud vision calls not permitted because allow_cloud_images=false"
            )


def _infer_cloud(provider: LLMProvider, decision: object | None) -> bool:
    if decision is not None and hasattr(decision, "cloud"):
        try:
            return bool(getattr(decision, "cloud"))
        except Exception:
            pass
    provider_type = type(provider).__name__
    if provider_type == "OpenAIProvider":
        return True
    base_url = getattr(provider, "_base_url", None)
    if base_url:
        parsed = urlparse(str(base_url))
        host = parsed.hostname or ""
        if host:
            return not is_loopback_host(host)
    return False
