"""Integration hooks for Memory Service."""

from __future__ import annotations

from typing import Iterable

from ..config import AppConfig
from ..logging_utils import get_logger
from .client import MemoryServiceClient
from .schemas import EntityHint, MemoryPolicyContext, MemoryQueryRequest

_LOG = get_logger("memory.hooks")


def fetch_memory_cards(
    config: AppConfig,
    *,
    query: str,
    client: MemoryServiceClient | None = None,
    namespace: str | None = None,
    entity_hints: Iterable[EntityHint] | None = None,
) -> tuple[list[dict], list[str]]:
    if not config.features.enable_memory_service_read_hook:
        return [], []
    if not config.memory_service.enabled:
        return [], ["memory_service_disabled"]
    memory_cfg = config.memory_service
    policy_config = memory_cfg.policy
    if not policy_config.allowed_audiences or not policy_config.sensitivity_order:
        return [], ["memory_service_policy_missing"]
    policy = MemoryPolicyContext(
        audience=list(policy_config.allowed_audiences),
        sensitivity_max=policy_config.sensitivity_order[-1],
    )
    request = MemoryQueryRequest(
        namespace=namespace or memory_cfg.default_namespace,
        query=query,
        policy=policy,
        entity_hints=list(entity_hints or []),
        max_cards=memory_cfg.retrieval.max_cards,
        max_tokens=memory_cfg.retrieval.max_tokens,
        topk_vector=memory_cfg.retrieval.topk_vector,
        topk_keyword=memory_cfg.retrieval.topk_keyword,
        topk_graph=memory_cfg.retrieval.topk_graph,
    )
    client = client or MemoryServiceClient(memory_cfg)
    try:
        response = client.query(request)
        cards = [card.model_dump(mode="json") for card in response.cards]
        return cards, response.warnings
    except Exception as exc:  # pragma: no cover - best effort
        _LOG.warning("Memory Service query failed: {}", exc)
        return [], ["memory_service_unavailable"]
