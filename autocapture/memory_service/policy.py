"""Policy validation for Memory Service ingestion and queries."""

from __future__ import annotations

import re
from typing import Iterable

from ..logging_utils import get_logger
from ..text.normalize import normalize_text
from .schemas import MemoryPolicyContext, MemoryProposal
from ..config import MemoryServicePolicyConfig

_LOG = get_logger("memory.policy")

_DEFAULT_PII_PATTERNS = [
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
]

_DEFAULT_SECRET_PATTERNS = [
    r"(?i)bearer\s+[A-Za-z0-9._+/=-]{8,}",
    r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9._+/=-]{6,}",
    r"(?i)token\s*[:=]\s*[A-Za-z0-9._+/=-]{6,}",
    r"(?i)secret\s*[:=]\s*[A-Za-z0-9._+/=-]{6,}",
    r"(?i)sk-[A-Za-z0-9-]{8,}",
    r"(?i)gh[pous]_[A-Za-z0-9-]{8,}",
]

_PREFERENCE_KEYS = {
    "preference",
    "preferences",
    "likes",
    "dislikes",
    "favorite",
    "favourite",
    "user_profile",
    "user_preferences",
}

_PERSON_KINDS = {"person", "user", "individual", "employee"}


class MemoryPolicyValidator:
    def __init__(self, config: MemoryServicePolicyConfig) -> None:
        self._config = config
        self._pii_patterns = _compile_patterns(
            config.pii_patterns if config.pii_patterns else _DEFAULT_PII_PATTERNS
        )
        self._secret_patterns = _compile_patterns(
            config.secret_patterns if config.secret_patterns else _DEFAULT_SECRET_PATTERNS
        )

    def validate_proposal(self, proposal: MemoryProposal) -> list[str]:
        reasons: list[str] = []
        if not proposal.content_text.strip():
            return ["content_empty"]
        if not proposal.policy:
            return ["policy_missing"]
        if not proposal.policy.audience:
            reasons.append("policy_audience_missing")
        if proposal.policy.sensitivity not in self._config.sensitivity_order:
            reasons.append("policy_sensitivity_invalid")
        if not _audiences_allowed(proposal.policy.audience, self._config.allowed_audiences):
            reasons.append("policy_audience_blocked")
        if _contains_patterns(proposal.content_text, self._pii_patterns):
            reasons.append("policy_pii_detected")
        if _contains_patterns(proposal.content_text, self._secret_patterns):
            reasons.append("policy_secret_detected")
        if self._config.reject_person_entities and _has_person_entity(proposal.entities):
            reasons.append("policy_person_entity_detected")
        if self._config.reject_preferences and _contains_preferences(proposal.content_json):
            reasons.append("policy_preference_detected")
        return reasons

    def validate_query_policy(self, policy: MemoryPolicyContext) -> list[str]:
        reasons: list[str] = []
        if not policy.audience:
            reasons.append("policy_audience_missing")
        if policy.sensitivity_max not in self._config.sensitivity_order:
            reasons.append("policy_sensitivity_invalid")
        if not _audiences_allowed(policy.audience, self._config.allowed_audiences):
            reasons.append("policy_audience_blocked")
        return reasons

    def policy_allows(
        self,
        *,
        audiences: Iterable[str],
        sensitivity_rank: int,
        policy: MemoryPolicyContext,
    ) -> bool:
        if not _audiences_allowed(policy.audience, self._config.allowed_audiences):
            return False
        if not _audiences_allowed(audiences, policy.audience):
            return False
        max_rank = self._config.sensitivity_order.index(policy.sensitivity_max)
        return sensitivity_rank <= max_rank


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            _LOG.warning("Invalid policy regex {}: {}", pattern, exc)
    return compiled


def _contains_patterns(text: str, patterns: list[re.Pattern[str]]) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in patterns)


def _has_person_entity(entities) -> bool:
    for entity in entities or []:
        kind = normalize_text(getattr(entity, "kind", "") or "").lower()
        if kind in _PERSON_KINDS:
            return True
    return False


def _contains_preferences(payload: object) -> bool:
    if payload is None:
        return False
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_norm = normalize_text(str(key)).lower()
            if key_norm in _PREFERENCE_KEYS:
                return True
            if _contains_preferences(value):
                return True
        return False
    if isinstance(payload, list):
        return any(_contains_preferences(item) for item in payload)
    if isinstance(payload, str):
        return any(token in normalize_text(payload).lower() for token in _PREFERENCE_KEYS)
    return False


def _audiences_allowed(requested: Iterable[str], allowed: Iterable[str]) -> bool:
    allowed_set = {normalize_text(item).lower() for item in allowed if item}
    requested_set = {normalize_text(item).lower() for item in requested if item}
    if not allowed_set or not requested_set:
        return False
    return bool(allowed_set & requested_set)
