"""API dependency helpers."""

from __future__ import annotations

from fastapi import Request

from .container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
