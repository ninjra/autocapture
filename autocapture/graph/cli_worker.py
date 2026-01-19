"""CLI bridge for graph worker adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import AppConfig
from .models import GraphIndexRequest, GraphQueryRequest
from .service import GraphService


def _load_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def _build_config(workspace_root: str | None) -> AppConfig:
    config = AppConfig()
    graph_cfg = config.graph_service.model_copy(update={"require_workers": False})
    if workspace_root:
        graph_cfg = graph_cfg.model_copy(update={"workspace_root": Path(workspace_root)})
    return config.model_copy(update={"graph_service": graph_cfg})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--mode", required=True, choices=["index", "query"])
    args = parser.parse_args()

    payload = _load_payload()
    workspace_root = payload.get("workspace_root") if isinstance(payload, dict) else None
    config = _build_config(workspace_root)
    service = GraphService(config)

    if args.mode == "index":
        request = GraphIndexRequest.model_validate(payload)
        response = service.index(request, adapter=args.adapter, use_workers=False)
    else:
        request = GraphQueryRequest.model_validate(payload)
        response = service.query(request, adapter=args.adapter, use_workers=False)

    sys.stdout.write(response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
