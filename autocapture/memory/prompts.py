"""Prompt registry and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir
        self._cache: dict[str, PromptTemplate] = {}

    def load(self) -> None:
        for path in self._prompts_dir.glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            name = payload["name"]
            template = PromptTemplate(
                name=name,
                version=payload["version"],
                system_prompt=payload["system_prompt"],
                tags=payload.get("tags", []),
                raw_template=payload.get("raw_template", payload["system_prompt"]),
                derived_template=payload.get("derived_template", payload["system_prompt"]),
            )
            self._cache[name] = template

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
