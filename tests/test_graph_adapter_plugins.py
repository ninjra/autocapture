from __future__ import annotations

from autocapture.config import GraphAdaptersConfig
from autocapture.memory.graph_adapters import GraphAdapterClient, GraphAdapterGroup


class _StubAdapter:
    def __init__(self) -> None:
        self.enabled = True
        self.calls: list[tuple[str, int]] = []

    def query(self, query: str, *, limit: int, time_range, filters):
        self.calls.append((query, limit))
        return []


class _StubPlugins:
    def __init__(self, adapter: _StubAdapter) -> None:
        self._adapter = adapter
        self.calls: list[tuple[str, str, dict]] = []

    def resolve_extension(self, kind: str, extension_id: str, **kwargs):
        self.calls.append((kind, extension_id, kwargs))
        return self._adapter


class _FailingPlugins:
    def resolve_extension(self, kind: str, extension_id: str, **kwargs):
        raise RuntimeError("boom")


def test_graph_adapter_group_uses_plugin_manager() -> None:
    config = GraphAdaptersConfig()
    adapter = _StubAdapter()
    plugins = _StubPlugins(adapter)
    group = GraphAdapterGroup(config, plugin_manager=plugins)

    group.query("q", limit=5, time_range=None, filters=None)

    kinds = [call[0] for call in plugins.calls]
    ids = [call[1] for call in plugins.calls]
    assert kinds == ["graph.adapter", "graph.adapter", "graph.adapter"]
    assert ids == ["graphrag", "hypergraphrag", "hyperrag"]
    assert len(adapter.calls) == 3


def test_graph_adapter_group_falls_back_on_plugin_error() -> None:
    config = GraphAdaptersConfig()
    group = GraphAdapterGroup(config, plugin_manager=_FailingPlugins())

    assert isinstance(group._adapters["graphrag"], GraphAdapterClient)
    assert isinstance(group._adapters["hypergraphrag"], GraphAdapterClient)
    assert isinstance(group._adapters["hyperrag"], GraphAdapterClient)
