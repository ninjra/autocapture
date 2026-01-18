"""Minimal module registry for optional runtime components."""

from __future__ import annotations

from typing import Iterable, Protocol


class AppModule(Protocol):
    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def health(self) -> dict: ...


class ModuleHost:
    def __init__(self, modules: Iterable[AppModule]) -> None:
        self._modules = list(modules)

    def start(self) -> None:
        for module in self._modules:
            module.start()

    def stop(self) -> None:
        for module in reversed(self._modules):
            module.stop()

    def health(self) -> dict[str, dict]:
        return {module.name: module.health() for module in self._modules}
