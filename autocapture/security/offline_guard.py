"""Offline egress guard for local-only mode."""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import Iterable

_lock = threading.Lock()
_enabled = False
_allowed_hosts: set[str] = set()
_original_connect = None
_original_connect_ex = None
_original_getaddrinfo = None


def is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return False


def _is_allowed_host(host: str) -> bool:
    if host in _allowed_hosts:
        return True
    if is_loopback(host):
        return True
    return False


def _extract_host(address) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    if isinstance(address, str):
        return address
    return None


def apply_offline_guard(allowed_hosts: Iterable[str], enabled: bool) -> None:
    """Monkeypatch socket APIs to block non-loopback egress."""

    global _enabled, _allowed_hosts, _original_connect, _original_connect_ex, _original_getaddrinfo

    if not enabled:
        return

    with _lock:
        _allowed_hosts = set(allowed_hosts)
        if _enabled:
            return
        _enabled = True

        _original_connect = socket.socket.connect
        _original_connect_ex = socket.socket.connect_ex
        _original_getaddrinfo = socket.getaddrinfo

        def guarded_connect(self, address):  # type: ignore[no-untyped-def]
            host = _extract_host(address)
            if host and not _is_allowed_host(host):
                raise RuntimeError("Offline mode enabled; outbound network blocked")
            return _original_connect(self, address)

        def guarded_connect_ex(self, address):  # type: ignore[no-untyped-def]
            host = _extract_host(address)
            if host and not _is_allowed_host(host):
                raise RuntimeError("Offline mode enabled; outbound network blocked")
            return _original_connect_ex(self, address)

        def guarded_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
            if host and not _is_allowed_host(str(host)):
                raise RuntimeError("Offline mode enabled; DNS resolution blocked")
            return _original_getaddrinfo(host, *args, **kwargs)

        socket.socket.connect = guarded_connect  # type: ignore[assignment]
        socket.socket.connect_ex = guarded_connect_ex  # type: ignore[assignment]
        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
