"""Prompt registry and persistence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Optional

import importlib.resources as resources

import yaml

from sqlalchemy import select

from ..storage.database import DatabaseManager
from ..logging_utils import get_logger
from ..storage.models import PromptLibraryRecord
from ..security.template_lint import lint_template_text
from ..paths import resource_root
from ..config import is_dev_mode


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system_prompt: str
    tags: list[str]
    raw_template: str
    derived_template: str


class PromptRegistry:
    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        package: Optional[str] = None,
        *,
        hardening_enabled: bool = True,
        log_provenance: bool = True,
        extra_dirs: list[Path] | None = None,
        allow_external: bool = False,
    ) -> None:
        self._prompts_dir = prompts_dir
        self._package = package
        self._extra_dirs = list(extra_dirs or [])
        self._allow_external = allow_external
        self._cache: dict[str, PromptTemplate] = {}
        self._hardening_enabled = hardening_enabled
        self._log_provenance = log_provenance
        self._log = get_logger("prompts")

    def load(self) -> None:
        if self._prompts_dir and self._prompts_dir.exists():
            if not _is_trusted_prompt_path(self._prompts_dir):
                raise ValueError(f"Untrusted prompt path: {self._prompts_dir}")
            for path in self._prompts_dir.glob("*.yaml"):
                raw = path.read_text(encoding="utf-8")
                template = _parse_prompt(raw, hardening_enabled=self._hardening_enabled)
                self._cache[template.name] = template
                self._maybe_log_provenance(
                    template, source=str(path), raw=raw, version=template.version
                )
        if self._package:
            for entry in resources.files(self._package).iterdir():
                if entry.name.endswith(".yaml"):
                    payload = entry.read_text(encoding="utf-8")
                    template = _parse_prompt(payload, hardening_enabled=self._hardening_enabled)
                    self._cache[template.name] = template
                    self._maybe_log_provenance(
                        template,
                        source=f"{self._package}:{entry.name}",
                        raw=payload,
                        version=template.version,
                    )
        for extra_dir in self._extra_dirs:
            if not extra_dir or not extra_dir.exists():
                continue
            if not self._allow_external and not _is_trusted_prompt_path(extra_dir):
                raise ValueError(f"Untrusted prompt path: {extra_dir}")
            for path in extra_dir.glob("*.yaml"):
                raw = path.read_text(encoding="utf-8")
                template = _parse_prompt(raw, hardening_enabled=self._hardening_enabled)
                self._cache[template.name] = template
                self._maybe_log_provenance(
                    template, source=str(path), raw=raw, version=template.version
                )

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
    def from_package(
        cls,
        package: str,
        *,
        hardening_enabled: bool = True,
        log_provenance: bool = True,
        extra_dirs: list[Path] | None = None,
        allow_external: bool = False,
    ) -> "PromptRegistry":
        return cls(
            package=package,
            hardening_enabled=hardening_enabled,
            log_provenance=log_provenance,
            extra_dirs=extra_dirs,
            allow_external=allow_external,
        )

    def _maybe_log_provenance(
        self, template: PromptTemplate, *, source: str, raw: str, version: str
    ) -> None:
        if not self._log_provenance:
            return
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        self._log.info(
            "Loaded prompt template {}:{} from {} (sha256={})",
            template.name,
            version,
            source,
            digest,
        )


def _parse_prompt(raw: str, *, hardening_enabled: bool) -> PromptTemplate:
    payload = yaml.safe_load(raw)
    name = payload["name"]
    system_prompt = payload["system_prompt"]
    raw_template = payload.get("raw_template", payload["system_prompt"])
    derived_template = payload.get("derived_template", payload["system_prompt"])
    if hardening_enabled:
        lint_template_text(system_prompt, label=f"prompt:{name}:system_prompt")
        lint_template_text(raw_template, label=f"prompt:{name}:raw_template")
        lint_template_text(derived_template, label=f"prompt:{name}:derived_template")
    return PromptTemplate(
        name=name,
        version=payload["version"],
        system_prompt=system_prompt,
        tags=payload.get("tags", []),
        raw_template=raw_template,
        derived_template=derived_template,
    )


def _is_trusted_prompt_path(path: Path) -> bool:
    if is_dev_mode():
        return True
    root = resource_root().resolve()
    try:
        path = path.resolve()
    except Exception:
        return False
    return root in path.parents or path == root


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
