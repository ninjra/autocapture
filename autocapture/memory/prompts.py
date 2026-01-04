"""Prompt registry and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import importlib.resources as resources

import yaml

from sqlalchemy import select

from ..storage.database import DatabaseManager
from ..storage.models import PromptLibraryRecord


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system_prompt: str
    tags: list[str]
    raw_template: str
    derived_template: str


class PromptRegistry:
    def __init__(self, prompts_dir: Optional[Path] = None, package: Optional[str] = None) -> None:
        self._prompts_dir = prompts_dir
        self._package = package
        self._cache: dict[str, PromptTemplate] = {}

    def load(self) -> None:
        if self._prompts_dir and self._prompts_dir.exists():
            for path in self._prompts_dir.glob("*.yaml"):
                template = _parse_prompt(path.read_text(encoding="utf-8"))
                self._cache[template.name] = template
        if self._package:
            for entry in resources.files(self._package).iterdir():
                if entry.name.endswith(".yaml"):
                    payload = entry.read_text(encoding="utf-8")
                    template = _parse_prompt(payload)
                    self._cache[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        if not self._cache:
            self.load()
        if name not in self._cache:
            raise KeyError(f"Prompt {name} not found")
        return self._cache[name]

    def all(self) -> Iterable[PromptTemplate]:
        if not self._cache:
            self.load()
        return self._cache.values()

    @classmethod
    def from_package(cls, package: str) -> "PromptRegistry":
        return cls(package=package)


def _parse_prompt(raw: str) -> PromptTemplate:
    payload = yaml.safe_load(raw)
    name = payload["name"]
    return PromptTemplate(
        name=name,
        version=payload["version"],
        system_prompt=payload["system_prompt"],
        tags=payload.get("tags", []),
        raw_template=payload.get("raw_template", payload["system_prompt"]),
        derived_template=payload.get("derived_template", payload["system_prompt"]),
    )


class PromptLibraryService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def sync_registry(self, registry: PromptRegistry) -> None:
        with self._db.session() as session:
            for prompt in registry.all():
                existing = session.execute(
                    select(PromptLibraryRecord).where(
                        PromptLibraryRecord.name == prompt.name,
                        PromptLibraryRecord.version == prompt.version,
                    )
                ).scalar_one_or_none()
                if existing:
                    continue
                session.add(
                    PromptLibraryRecord(
                        name=prompt.name,
                        version=prompt.version,
                        raw_template=prompt.raw_template,
                        derived_template=prompt.derived_template,
                        tags=prompt.tags,
                    )
                )
