"""Graph worker CLI wrappers for GraphRAG/HyperGraphRAG/Hyper-RAG."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import GraphIndexRequest, GraphIndexResponse, GraphQueryRequest, GraphQueryResponse


@dataclass(frozen=True)
class GraphWorkerSpec:
    name: str
    cli_path: str
    timeout_s: float

    def enabled(self) -> bool:
        return bool(self.cli_path)


class GraphWorker:
    def __init__(self, spec: GraphWorkerSpec, *, workspace_root: Path) -> None:
        self._spec = spec
        self._workspace_root = workspace_root

    def _run(self, payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
        if not self._spec.enabled():
            raise RuntimeError(f"{self._spec.name}_cli_missing")
        cmd = [self._spec.cli_path, "--adapter", self._spec.name, "--mode", mode]
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=self._spec.timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"{self._spec.name}_cli_failed")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_worker_response") from exc
        if not isinstance(data, dict):
            raise RuntimeError("invalid_worker_response")
        return data

    def index(self, request: GraphIndexRequest) -> GraphIndexResponse:
        payload = request.model_dump(mode="json")
        payload["workspace_root"] = str(self._workspace_root)
        data = self._run(payload, mode="index")
        return GraphIndexResponse.model_validate(data)

    def query(self, request: GraphQueryRequest) -> GraphQueryResponse:
        payload = request.model_dump(mode="json")
        payload["workspace_root"] = str(self._workspace_root)
        data = self._run(payload, mode="query")
        return GraphQueryResponse.model_validate(data)


class GraphWorkerGroup:
    def __init__(self, specs: list[GraphWorkerSpec], *, workspace_root: Path) -> None:
        self._workers = {
            spec.name: GraphWorker(spec, workspace_root=workspace_root) for spec in specs
        }

    def index(self, adapter: str, request: GraphIndexRequest) -> GraphIndexResponse:
        worker = self._workers.get(adapter)
        if worker is None:
            raise RuntimeError("adapter_not_supported")
        return worker.index(request)

    def query(self, adapter: str, request: GraphQueryRequest) -> GraphQueryResponse:
        worker = self._workers.get(adapter)
        if worker is None:
            raise RuntimeError("adapter_not_supported")
        return worker.query(request)


__all__ = ["GraphWorker", "GraphWorkerGroup", "GraphWorkerSpec"]
