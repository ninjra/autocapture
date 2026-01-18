from __future__ import annotations

from autocapture.config import AppConfig
from autocapture.runtime_env import ProfileName
from autocapture.runtime_profile import ProfileScheduler


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


def test_profile_scheduler_params() -> None:
    config = AppConfig()
    scheduler = ProfileScheduler(config)
    foreground = scheduler.profile(ProfileName.FOREGROUND)
    idle = scheduler.profile(ProfileName.IDLE)

    assert idle.ocr_workers >= foreground.ocr_workers
    assert idle.embed_workers >= foreground.embed_workers
    assert idle.agent_workers >= foreground.agent_workers
    assert _cmp_optional(idle.ocr_batch_size, foreground.ocr_batch_size) >= 0
    assert _cmp_optional(idle.embed_batch_size, foreground.embed_batch_size) >= 0
    assert _cmp_optional(idle.reranker_batch_size, foreground.reranker_batch_size) >= 0
    assert idle.poll_interval_s >= foreground.poll_interval_s
