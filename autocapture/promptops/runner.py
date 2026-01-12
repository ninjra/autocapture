"""PromptOps ingestion, proposal generation, evaluation, and PR automation."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx
import yaml

from ..config import AppConfig
from ..evals import EvalMetrics, run_eval
from ..logging_utils import get_logger
from ..memory.router import ProviderRouter
from ..resilience import (
    CircuitBreaker,
    RetryPolicy,
    is_retryable_exception,
    is_retryable_http_status,
    retry_sync,
)
from ..storage.database import DatabaseManager
from ..storage.models import PromptOpsRunRecord


@dataclass(frozen=True)
class SourceSnapshot:
    source: str
    fetched_at: str
    status: str
    sha256: str
    path: str
    error: Optional[str]
    is_local: bool
    excerpt: str


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    system_prompt: str
    raw_template: str
    derived_template: str
    tags: list[str]
    rationale: str = ""


@dataclass(frozen=True)
class PromptProposal:
    name: str
    raw_path: Path
    derived_path: Path
    raw_content: str
    derived_content: str
    rationale: str


class PromptOpsRunner:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        *,
        llm_provider=None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._log = get_logger("promptops")
        self._retry_policy = RetryPolicy()
        self._breaker = CircuitBreaker()
        self._llm_provider = llm_provider
        self._http = http_client

    def run_once(self, sources: Iterable[str] | None = None) -> PromptOpsRunRecord:
        promptops = self._config.promptops
        source_list = list(sources or promptops.sources)
        run = PromptOpsRunRecord(
            sources_fetched={},
            proposals={},
            eval_results={},
            status="started",
        )
        with self._db.session() as session:
            session.add(run)
            session.flush()
            run_id = run.run_id

        try:
            fetched = self._fetch_sources(run_id, source_list)
            self._update_run(run_id, sources_fetched=[item.__dict__ for item in fetched])
            self._update_run(run_id, status="fetched")

            proposals = self._propose_prompts(fetched)
            proposal_payload = {
                item.name: {
                    "raw_path": str(item.raw_path),
                    "derived_path": str(item.derived_path),
                    "rationale": item.rationale,
                    "diff_raw": _diff_strings(item.raw_path, item.raw_content),
                    "diff_derived": _diff_strings(item.derived_path, item.derived_content),
                }
                for item in proposals
            }
            self._update_run(run_id, proposals=proposal_payload, status="proposed")

            baseline = run_eval(self._config, Path("evals/golden_queries.json"))
            proposed = run_eval(
                self._config,
                Path("evals/golden_queries.json"),
                overrides=proposals,
            )
            acceptance = self._should_accept(baseline, proposed)
            eval_results = {
                "baseline": baseline.to_dict(),
                "proposed": proposed.to_dict(),
                "accepted": acceptance,
                "tolerance": promptops.acceptance_tolerance,
            }
            self._update_run(run_id, eval_results=eval_results, status="evaluated")

            if acceptance:
                pr_url = self._maybe_open_pr(run_id, proposals, fetched, eval_results)
                if pr_url:
                    self._update_run(run_id, pr_url=pr_url, status="pr_opened")
                else:
                    self._update_run(run_id, status="completed_no_pr")
            else:
                self._log.warning("PromptOps proposal rejected by eval gate")
                self._update_run(run_id, status="skipped")
        except Exception as exc:
            self._log.warning("PromptOps run failed: {}", exc)
            self._update_run(run_id, status="failed")
        with self._db.session() as session:
            return session.get(PromptOpsRunRecord, run_id)

    def _fetch_sources(self, run_id: str, sources: list[str]) -> list[SourceSnapshot]:
        snapshots: list[SourceSnapshot] = []
        if not sources:
            self._log.warning("PromptOps has no sources configured.")
            return snapshots
        promptops_dir = Path(self._config.capture.data_dir) / "promptops"
        promptops_dir.mkdir(parents=True, exist_ok=True)
        source_dir = promptops_dir / "sources" / run_id
        source_dir.mkdir(parents=True, exist_ok=True)
        for idx, source in enumerate(sources, start=1):
            parsed = urlparse(source)
            is_local = parsed.scheme in ("", "file")
            fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
            if not is_local and self._config.offline and not self._config.privacy.cloud_enabled:
                snapshots.append(
                    SourceSnapshot(
                        source=source,
                        fetched_at=fetched_at,
                        status="blocked_offline",
                        sha256="",
                        path="",
                        error="offline_mode",
                        is_local=False,
                        excerpt="",
                    )
                )
                continue
            try:
                body = self._load_source(source, is_local)
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                filename = f"{idx:02d}_{digest[:12]}.txt"
                path = source_dir / filename
                path.write_text(body, encoding="utf-8")
                snapshots.append(
                    SourceSnapshot(
                        source=source,
                        fetched_at=fetched_at,
                        status="ok",
                        sha256=digest,
                        path=str(path),
                        error=None,
                        is_local=is_local,
                        excerpt=body[:2000],
                    )
                )
            except Exception as exc:
                snapshots.append(
                    SourceSnapshot(
                        source=source,
                        fetched_at=fetched_at,
                        status="error",
                        sha256="",
                        path="",
                        error=str(exc),
                        is_local=is_local,
                        excerpt="",
                    )
                )
        return snapshots

    def _load_source(self, source: str, is_local: bool) -> str:
        if is_local:
            path = Path(source.replace("file://", ""))
            return path.read_text(encoding="utf-8")

        if not self._breaker.allow():
            raise RuntimeError("PromptOps breaker open; skipping fetch")

        def _fetch() -> httpx.Response:
            if self._http:
                response = self._http.get(source)
            else:
                with httpx.Client(timeout=20.0) as client:
                    response = client.get(source)
            if is_retryable_http_status(response.status_code):
                raise httpx.HTTPStatusError(
                    "Retryable status",
                    request=response.request,
                    response=response,
                )
            return response

        response = retry_sync(
            _fetch,
            policy=self._retry_policy,
            is_retryable=is_retryable_exception,
        )
        self._breaker.record_success()
        return response.text

    def _propose_prompts(self, sources: list[SourceSnapshot]) -> list[PromptProposal]:
        prompts = _load_prompt_specs(Path("prompts/raw"))
        proposals: list[PromptProposal] = []
        for prompt in prompts:
            proposal = self._generate_prompt_proposal(prompt, sources)
            proposals.append(proposal)
        return proposals

    def _generate_prompt_proposal(
        self, prompt: PromptSpec, sources: list[SourceSnapshot]
    ) -> PromptProposal:
        llm = self._llm_provider
        if llm is None:
            llm = ProviderRouter(
                self._config.routing,
                self._config.llm,
                offline=self._config.offline,
                privacy=self._config.privacy,
            ).select_llm()[0]
        system_prompt, query, context = _promptops_instruction(prompt, sources)
        response = asyncio.run(llm.generate_answer(system_prompt, query, context))
        spec = _parse_promptops_response(response, prompt)
        raw_path = Path("prompts/raw") / f"{prompt.name.lower()}.yaml"
        derived_path = Path("autocapture/prompts/derived") / f"{prompt.name.lower()}.yaml"
        raw_content = _serialize_prompt(spec)
        derived_content = _serialize_prompt(
            PromptSpec(
                name=spec.name,
                version=spec.version,
                system_prompt=spec.system_prompt,
                raw_template=spec.raw_template,
                derived_template=spec.derived_template,
                tags=spec.tags,
                rationale=spec.rationale,
            )
        )
        return PromptProposal(
            name=spec.name,
            raw_path=raw_path,
            derived_path=derived_path,
            raw_content=raw_content,
            derived_content=derived_content,
            rationale=spec.rationale,
        )

    def _should_accept(self, baseline: EvalMetrics, proposed: EvalMetrics) -> bool:
        tolerance = self._config.promptops.acceptance_tolerance
        return proposed.verifier_pass_rate >= baseline.verifier_pass_rate - tolerance

    def _maybe_open_pr(
        self,
        run_id: str,
        proposals: list[PromptProposal],
        sources: list[SourceSnapshot],
        eval_results: dict,
    ) -> Optional[str]:
        promptops = self._config.promptops
        if not promptops.github_token or not promptops.github_repo:
            self._write_patch(run_id, proposals)
            return None
        return _open_github_pr(
            promptops.github_repo,
            promptops.github_token,
            proposals,
            sources,
            eval_results,
            http_client=self._http,
        )

    def _write_patch(self, run_id: str, proposals: list[PromptProposal]) -> None:
        promptops_dir = Path(self._config.capture.data_dir) / "promptops"
        patch_dir = promptops_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        diff_chunks = []
        for proposal in proposals:
            diff_chunks.append(_diff_strings(proposal.raw_path, proposal.raw_content))
            diff_chunks.append(_diff_strings(proposal.derived_path, proposal.derived_content))
        patch_path = patch_dir / f"promptops_{run_id}.diff"
        patch_path.write_text("\n".join(diff_chunks), encoding="utf-8")

    def _update_run(self, run_id: str, **fields) -> None:
        with self._db.session() as session:
            run = session.get(PromptOpsRunRecord, run_id)
            if not run:
                return
            for key, value in fields.items():
                setattr(run, key, value)


def _promptops_instruction(
    prompt: PromptSpec, sources: list[SourceSnapshot]
) -> tuple[str, str, str]:
    system = (
        "You are PromptOps. Propose an updated prompt YAML. "
        "Return YAML with keys: name, version, system_prompt, raw_template, "
        "derived_template, tags, rationale."
    )
    query = f"Update prompt {prompt.name} based on sources."
    sources_payload = [
        {
            "source": item.source,
            "sha256": item.sha256,
            "status": item.status,
            "path": item.path,
            "excerpt": item.excerpt,
        }
        for item in sources
    ]
    context = json.dumps(
        {
            "current_prompt": _prompt_to_dict(prompt),
            "sources": sources_payload,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, query, context


def _parse_promptops_response(response: str, current: PromptSpec) -> PromptSpec:
    payload = _extract_yaml(response)
    data = yaml.safe_load(payload) if payload else {}
    if not isinstance(data, dict):
        raise ValueError("PromptOps response invalid: expected YAML mapping")
    name = data.get("name") or current.name
    if name != current.name:
        raise ValueError("PromptOps response changed prompt name unexpectedly")
    version = data.get("version") or _increment_version(current.version)
    if version == current.version:
        version = _increment_version(current.version)
    system_prompt = data.get("system_prompt") or current.system_prompt
    raw_template = data.get("raw_template") or system_prompt
    derived_template = data.get("derived_template") or raw_template
    tags = data.get("tags") or current.tags
    if not isinstance(tags, list):
        raise ValueError("PromptOps tags must be a list")
    rationale = data.get("rationale") or ""
    return PromptSpec(
        name=name,
        version=version,
        system_prompt=system_prompt,
        raw_template=raw_template,
        derived_template=derived_template,
        tags=tags,
        rationale=rationale,
    )


def _extract_yaml(response: str) -> str:
    if "```" not in response:
        return response.strip()
    parts = response.split("```")
    for idx, part in enumerate(parts):
        if part.strip().startswith("yaml"):
            return parts[idx + 1].strip()
    return parts[1].strip() if len(parts) > 1 else response.strip()


def _increment_version(version: str) -> str:
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    return f"{version}-1"


def _serialize_prompt(spec: PromptSpec) -> str:
    payload = _prompt_to_dict(spec)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _prompt_to_dict(spec: PromptSpec) -> dict:
    return {
        "name": spec.name,
        "version": spec.version,
        "system_prompt": spec.system_prompt,
        "raw_template": spec.raw_template,
        "derived_template": spec.derived_template,
        "tags": spec.tags,
        "rationale": spec.rationale,
    }


def _load_prompt_specs(directory: Path) -> list[PromptSpec]:
    specs: list[PromptSpec] = []
    for path in directory.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        specs.append(
            PromptSpec(
                name=data["name"],
                version=data["version"],
                system_prompt=data["system_prompt"],
                raw_template=data.get("raw_template", data["system_prompt"]),
                derived_template=data.get("derived_template", data["system_prompt"]),
                tags=data.get("tags", []),
            )
        )
    return specs


def _diff_strings(path: Path, new_content: str) -> str:
    try:
        old_content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        old_content = ""
    diff = difflib.unified_diff(
        old_content.splitlines(),
        new_content.splitlines(),
        fromfile=str(path),
        tofile=str(path),
        lineterm="",
    )
    return "\n".join(diff)


def _open_github_pr(
    repo: str,
    token: str,
    proposals: list[PromptProposal],
    sources: list[SourceSnapshot],
    eval_results: dict,
    *,
    http_client: httpx.Client | None = None,
) -> str:
    owner, repo_name = repo.split("/", 1)
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    client = http_client or httpx.Client(timeout=10.0)
    repo_info = client.get(f"https://api.github.com/repos/{owner}/{repo_name}", headers=headers)
    repo_info.raise_for_status()
    default_branch = repo_info.json()["default_branch"]
    ref_resp = client.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/ref/heads/{default_branch}",
        headers=headers,
    )
    ref_resp.raise_for_status()
    base_sha = ref_resp.json()["object"]["sha"]
    branch_name = f"promptops/{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')}-{base_sha[:7]}"
    client.post(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
    ).raise_for_status()

    for proposal in proposals:
        _update_github_file(
            client,
            headers,
            owner,
            repo_name,
            proposal.raw_path,
            proposal.raw_content,
            branch_name,
        )
        _update_github_file(
            client,
            headers,
            owner,
            repo_name,
            proposal.derived_path,
            proposal.derived_content,
            branch_name,
        )

    title = f"PromptOps: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')} updates"
    body = _render_pr_body(sources, proposals, eval_results)
    pr_resp = client.post(
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls",
        headers=headers,
        json={
            "title": title,
            "head": branch_name,
            "base": default_branch,
            "body": body,
        },
    )
    pr_resp.raise_for_status()
    return pr_resp.json()["html_url"]


def _update_github_file(
    client: httpx.Client,
    headers: dict,
    owner: str,
    repo: str,
    path: Path,
    content: str,
    branch: str,
) -> None:
    path_str = str(path).replace("\\", "/")
    get_resp = client.get(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path_str}",
        headers=headers,
        params={"ref": branch},
    )
    sha = None
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")
    payload = {
        "message": f"PromptOps update {path_str}",
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    put_resp = client.put(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path_str}",
        headers=headers,
        json=payload,
    )
    put_resp.raise_for_status()


def _render_pr_body(
    sources: list[SourceSnapshot],
    proposals: list[PromptProposal],
    eval_results: dict,
) -> str:
    lines = ["## PromptOps run", "", "### Sources"]
    for item in sources:
        lines.append(f"- {item.source} ({item.status}, sha={item.sha256[:8]})")
    lines.append("")
    lines.append("### Prompt changes")
    for proposal in proposals:
        lines.append(f"- {proposal.name}: {proposal.raw_path} + {proposal.derived_path}")
        if proposal.rationale:
            lines.append(f"  - Rationale: {proposal.rationale}")
    lines.append("")
    lines.append("### Eval results")
    lines.append(json.dumps(eval_results, indent=2))
    lines.append("")
    lines.append("### Rollback")
    lines.append("Revert this PR to restore the previous prompt versions.")
    return "\n".join(lines)
