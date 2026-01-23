"""Execution profile scheduler for foreground vs idle modes."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from .config import AppConfig, RuntimeQosProfile
from .runtime_env import ProfileName, ProfileTuning, RuntimeEnvConfig


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    name: ProfileName
    qos_profile: RuntimeQosProfile
    poll_interval_s: float
    ocr_workers: int
    embed_workers: int
    agent_workers: int
    ocr_batch_size: int | None
    embed_batch_size: int | None
    reranker_batch_size: int | None
    queue_depth_limit: int
    backpressure_threshold: int


class ProfileScheduler:
    def __init__(self, config: AppConfig, runtime_env: RuntimeEnvConfig | None = None) -> None:
        self._config = config
        self._runtime_env = runtime_env
        self._log = logging.getLogger("runtime.profile")
        base_poll = max(0.1, float(config.worker.poll_interval_s))
        foreground_poll = base_poll
        idle_poll = max(foreground_poll * 1.5, foreground_poll)

        foreground_qos = config.runtime.qos.profile_active
        idle_qos = config.runtime.qos.profile_idle

        self._foreground = self._build_profile(
            ProfileName.FOREGROUND, foreground_qos, foreground_poll
        )
        self._idle = self._build_profile(ProfileName.IDLE, idle_qos, idle_poll)
        self._idle = self._normalize_idle(self._foreground, self._idle)
        if runtime_env is not None:
            self._foreground_tuning = runtime_env.foreground_tuning
            self._idle_tuning = runtime_env.idle_tuning
        else:
            self._foreground_tuning = _fallback_tuning(ProfileName.FOREGROUND)
            self._idle_tuning = _fallback_tuning(ProfileName.IDLE)

    def profile(self, name: ProfileName) -> ExecutionProfile:
        return self._idle if name == ProfileName.IDLE else self._foreground

    def qos_profile(self, name: ProfileName) -> RuntimeQosProfile:
        return self.profile(name).qos_profile

    def tuning(self, name: ProfileName | None = None) -> ProfileTuning:
        if name is None and self._runtime_env is not None:
            name = self._runtime_env.profile
        if name == ProfileName.IDLE:
            return self._idle_tuning
        return self._foreground_tuning

    def max_workers(self, name: ProfileName | None = None) -> int:
        return self.tuning(name).max_workers

    def batch_size(self, name: ProfileName | None = None) -> int:
        return self.tuning(name).batch_size

    def poll_interval_s(self, name: ProfileName | None = None) -> float:
        return self.tuning(name).poll_interval_ms / 1000.0

    def sleep_interval(self, name: ProfileName | None = None) -> float:
        return self.poll_interval_s(name)

    def max_queue_depth(self, name: ProfileName | None = None) -> int:
        return self.tuning(name).max_queue_depth

    def max_cpu_pct_hint(self, name: ProfileName | None = None) -> int:
        return self.tuning(name).max_cpu_pct_hint

    def _build_profile(
        self,
        name: ProfileName,
        qos_profile: RuntimeQosProfile,
        poll_interval_s: float,
    ) -> ExecutionProfile:
        return ExecutionProfile(
            name=name,
            qos_profile=qos_profile,
            poll_interval_s=poll_interval_s,
            ocr_workers=int(qos_profile.ocr_workers),
            embed_workers=int(qos_profile.embed_workers),
            agent_workers=int(qos_profile.agent_workers),
            ocr_batch_size=qos_profile.ocr_batch_size,
            embed_batch_size=qos_profile.embed_batch_size,
            reranker_batch_size=qos_profile.reranker_batch_size,
            queue_depth_limit=int(self._config.capture.max_pending),
            backpressure_threshold=int(self._config.worker.ocr_backlog_soft_limit),
        )

    def _normalize_idle(
        self, foreground: ExecutionProfile, idle: ExecutionProfile
    ) -> ExecutionProfile:
        updates: dict[str, object] = {}
        if idle.ocr_workers < foreground.ocr_workers:
            updates["ocr_workers"] = foreground.ocr_workers
        if idle.embed_workers < foreground.embed_workers:
            updates["embed_workers"] = foreground.embed_workers
        if idle.agent_workers < foreground.agent_workers:
            updates["agent_workers"] = foreground.agent_workers
        if _cmp_optional(idle.ocr_batch_size, foreground.ocr_batch_size) < 0:
            updates["ocr_batch_size"] = foreground.ocr_batch_size
        if _cmp_optional(idle.embed_batch_size, foreground.embed_batch_size) < 0:
            updates["embed_batch_size"] = foreground.embed_batch_size
        if _cmp_optional(idle.reranker_batch_size, foreground.reranker_batch_size) < 0:
            updates["reranker_batch_size"] = foreground.reranker_batch_size
        if not updates:
            return idle
        self._log.warning("Idle profile below foreground; applying monotonic adjustments")
        qos_updates = {
            "ocr_workers": updates.get("ocr_workers", idle.ocr_workers),
            "embed_workers": updates.get("embed_workers", idle.embed_workers),
            "agent_workers": updates.get("agent_workers", idle.agent_workers),
            "ocr_batch_size": updates.get("ocr_batch_size", idle.ocr_batch_size),
            "embed_batch_size": updates.get("embed_batch_size", idle.embed_batch_size),
            "reranker_batch_size": updates.get("reranker_batch_size", idle.reranker_batch_size),
        }
        idle_qos = _clone_qos_profile(idle.qos_profile, qos_updates)
        return ExecutionProfile(
            name=idle.name,
            qos_profile=idle_qos,
            poll_interval_s=idle.poll_interval_s,
            ocr_workers=int(qos_updates["ocr_workers"]),
            embed_workers=int(qos_updates["embed_workers"]),
            agent_workers=int(qos_updates["agent_workers"]),
            ocr_batch_size=qos_updates["ocr_batch_size"],
            embed_batch_size=qos_updates["embed_batch_size"],
            reranker_batch_size=qos_updates["reranker_batch_size"],
            queue_depth_limit=idle.queue_depth_limit,
            backpressure_threshold=idle.backpressure_threshold,
        )


def _clone_qos_profile(base: RuntimeQosProfile, updates: dict[str, object]) -> RuntimeQosProfile:
    if hasattr(base, "model_dump"):
        data = base.model_dump()
    else:
        data = base.dict()
    data.update(updates)
    return RuntimeQosProfile(**data)


def _cmp_optional(left: int | None, right: int | None) -> int:
    if left is None and right is None:
        return 0
    if left is None:
        return -1
    if right is None:
        return 1
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def _fallback_tuning(name: ProfileName) -> ProfileTuning:
    cpu_count = os.cpu_count() or 1
    if name == ProfileName.FOREGROUND:
        max_workers = max(1, min(4, cpu_count))
        return ProfileTuning(
            max_workers=max_workers,
            batch_size=8,
            poll_interval_ms=100,
            max_queue_depth=1000,
            max_cpu_pct_hint=15,
        )
    return ProfileTuning(
        max_workers=max(1, cpu_count),
        batch_size=32,
        poll_interval_ms=500,
        max_queue_depth=5000,
        max_cpu_pct_hint=80,
    )
