"""Local FastAPI server for search, answer, and dashboard views."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..config import AppConfig
from ..llm.providers import (
    ContextChunk,
    LLMProvider,
    OpenAIProvider,
    OllamaProvider,
    build_citations,
)
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, OCRSpanRecord


class SuggestRequest(BaseModel):
    q: str


class SearchRequest(BaseModel):
    q: str
    time_range: Optional[tuple[dt.datetime, dt.datetime]] = None


class AnswerRequest(BaseModel):
    q: str


class Suggestion(BaseModel):
    segment_id: str
    span_id: int
    snippet: str
    captured_at: dt.datetime


class SearchResult(BaseModel):
    segment_id: str
    span_id: int
    snippet: str
    captured_at: dt.datetime


class CitationPayload(BaseModel):
    segment_id: str
    ts_range: str
    snippets: list[str]
    thumbs: list[str]


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationPayload]
    answer_id: str
    answer_url: str


@dataclass
class AnswerRecord:
    answer: str
    citations: list[CitationPayload]
    created_at: dt.datetime


class AnswerCache:
    def __init__(self) -> None:
        self._answers: dict[str, AnswerRecord] = {}
        self._lock = asyncio.Lock()

    async def store(self, answer: AnswerRecord) -> str:
        answer_id = str(uuid.uuid4())
        async with self._lock:
            self._answers[answer_id] = answer
        return answer_id

    async def fetch(self, answer_id: str) -> AnswerRecord | None:
        async with self._lock:
            return self._answers.get(answer_id)


class SearchService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("api.search")

    def suggest(self, query: str, limit: int = 10) -> list[Suggestion]:
        if not query.strip():
            return []
        with self._db.session() as session:
            stmt = (
                select(OCRSpanRecord, CaptureRecord)
                .join(CaptureRecord, OCRSpanRecord.capture_id == CaptureRecord.id)
                .where(OCRSpanRecord.text.ilike(f"%{query}%"))
                .order_by(CaptureRecord.captured_at.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).all()
        suggestions: list[Suggestion] = []
        for span, capture in rows:
            suggestions.append(
                Suggestion(
                    segment_id=capture.id,
                    span_id=span.id,
                    snippet=span.text[:160],
                    captured_at=capture.captured_at,
                )
            )
        return suggestions

    def search(
        self, query: str, time_range: Optional[tuple[dt.datetime, dt.datetime]]
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        with self._db.session() as session:
            stmt = (
                select(OCRSpanRecord, CaptureRecord)
                .join(CaptureRecord, OCRSpanRecord.capture_id == CaptureRecord.id)
                .where(OCRSpanRecord.text.ilike(f"%{query}%"))
                .order_by(CaptureRecord.captured_at.desc())
                .limit(50)
            )
            if time_range:
                stmt = stmt.where(
                    CaptureRecord.captured_at.between(time_range[0], time_range[1])
                )
            rows = session.execute(stmt).all()
        results: list[SearchResult] = []
        for span, capture in rows:
            results.append(
                SearchResult(
                    segment_id=capture.id,
                    span_id=span.id,
                    snippet=span.text[:200],
                    captured_at=capture.captured_at,
                )
            )
        self._log.info("Search '%s' -> %d hits", query, len(results))
        return results

    def context(self, query: str, limit: int = 8) -> list[ContextChunk]:
        suggestions = self.suggest(query, limit=limit)
        context: list[ContextChunk] = []
        for suggestion in suggestions:
            ts_range = suggestion.captured_at.isoformat()
            context.append(
                ContextChunk(
                    segment_id=suggestion.segment_id,
                    ts_range=ts_range,
                    snippet=suggestion.snippet,
                )
            )
        return context


class LLMService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self._log = get_logger("api.answer")

    async def answer(self, query: str, context: list[ContextChunk]) -> str:
        if not context:
            return (
                "I could not find any matching captures. Try a different query or "
                "open the dashboard to browse the timeline."
            )
        try:
            answer = await self._provider.generate_answer(query, context)
        except Exception as exc:
            self._log.warning("LLM provider failed: %s", exc)
            return (
                "I found relevant captures but could not reach the language model. "
                "Please click a citation to inspect the timeline."
            )
        if "[" not in answer:
            citations = ", ".join(
                f"[{chunk.segment_id} @ {chunk.ts_range}]" for chunk in context[:3]
            )
            answer = f"{answer}\n\nCitations: {citations}"
        return answer


def create_app(
    config: AppConfig,
    db_manager: DatabaseManager | None = None,
    llm_provider: LLMProvider | None = None,
) -> FastAPI:
    app = FastAPI(title="Autocapture API")
    db = db_manager or DatabaseManager(config.database)
    provider = llm_provider or build_provider(config)
    search_service = SearchService(db)
    llm_service = LLMService(provider)
    answer_cache = AnswerCache()
    log = get_logger("api")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "queue_depth": 0,
            "worker_status": "idle",
        }

    @app.post("/api/suggest")
    async def suggest(request: SuggestRequest) -> list[Suggestion]:
        return search_service.suggest(request.q)

    @app.post("/api/search")
    async def search(request: SearchRequest) -> list[SearchResult]:
        return search_service.search(request.q, request.time_range)

    @app.post("/api/answer")
    async def answer(request: AnswerRequest) -> AnswerResponse:
        context = search_service.context(request.q)
        answer_text = await llm_service.answer(request.q, context)
        citations = [
            CitationPayload(
                segment_id=citation.segment_id,
                ts_range=citation.ts_range,
                snippets=citation.snippets,
                thumbs=citation.thumbs,
            )
            for citation in build_citations(context)
        ]
        answer_record = AnswerRecord(
            answer=answer_text,
            citations=citations,
            created_at=dt.datetime.now(dt.timezone.utc),
        )
        answer_id = await answer_cache.store(answer_record)
        answer_url = f"http://127.0.0.1:{config.api.port}/answer/{answer_id}"
        log.info("Answer generated for '%s'", request.q)
        return AnswerResponse(
            answer=answer_text,
            citations=citations,
            answer_id=answer_id,
            answer_url=answer_url,
        )

    @app.get("/answer/{answer_id}")
    async def answer_page(answer_id: str) -> HTMLResponse:
        record = await answer_cache.fetch(answer_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Answer not found")
        citation_items: list[str] = []
        for citation in record.citations:
            snippet_text = "; ".join(citation.snippets)
            citation_items.append(
                f"<li><a href='/segment/{citation.segment_id}'>"
                f"{citation.segment_id}</a> @ {citation.ts_range}<br>"
                f"<small>{snippet_text}</small></li>"
            )
        citations_html = "".join(citation_items)
        html = f"""
        <html>
        <head>
          <title>Autocapture Answer</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; background: #0b0d13; color: #f4f5f7; }}
            .card {{ background: #161b22; padding: 20px; border-radius: 16px; }}
            a {{ color: #8bd5ff; }}
          </style>
        </head>
        <body>
          <div class='card'>
            <h2>Answer</h2>
            <p>{record.answer}</p>
            <h3>Citations</h3>
            <ul>{citations_html}</ul>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    @app.get("/segment/{segment_id}")
    async def segment(segment_id: str) -> HTMLResponse:
        with db.session() as session:
            capture = session.get(CaptureRecord, segment_id)
            if capture is None:
                raise HTTPException(status_code=404, detail="Segment not found")
            stmt = (
                select(OCRSpanRecord)
                .where(OCRSpanRecord.capture_id == segment_id)
                .order_by(OCRSpanRecord.id.asc())
            )
            spans = session.execute(stmt).scalars().all()
        span_list = "".join(
            f"<li><strong>#{span.id}</strong> {span.text}</li>" for span in spans
        )
        html = f"""
        <html>
        <head>
          <title>Segment {capture.id}</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; background: #0b0d13; color: #f4f5f7; }}
            .card {{ background: #161b22; padding: 20px; border-radius: 16px; }}
            a {{ color: #8bd5ff; }}
          </style>
        </head>
        <body>
          <div class='card'>
            <h2>Segment {capture.id}</h2>
            <p><strong>Captured at:</strong> {capture.captured_at}</p>
            <p><strong>Foreground:</strong> {capture.foreground_process}</p>
            <p><strong>Window:</strong> {capture.foreground_window}</p>
            <p><strong>Image:</strong> {capture.image_path}</p>
            <p><a href='file:///{capture.image_path}'>Open image</a></p>
            <h3>OCR Timeline</h3>
            <ul>{span_list}</ul>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    @app.get("/dashboard")
    async def dashboard() -> HTMLResponse:
        with db.session() as session:
            stmt = (
                select(CaptureRecord)
                .order_by(CaptureRecord.captured_at.desc())
                .limit(100)
            )
            captures = session.execute(stmt).scalars().all()
        items = "".join(
            f"<li><a href='/segment/{cap.id}'>"
            f"{cap.captured_at} - {cap.foreground_window}</a></li>"
            for cap in captures
        )
        html = f"""
        <html>
        <head>
          <title>Autocapture Dashboard</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; background: #0b0d13; color: #f4f5f7; }}
            .card {{ background: #161b22; padding: 20px; border-radius: 16px; }}
            a {{ color: #8bd5ff; }}
          </style>
        </head>
        <body>
          <div class='card'>
            <h2>Recent Captures</h2>
            <ul>{items}</ul>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    return app


def build_provider(config: AppConfig) -> LLMProvider:
    if config.llm.provider == "openai" and config.llm.openai_api_key:
        return OpenAIProvider(config.llm.openai_api_key, config.llm.openai_model)
    return OllamaProvider(config.llm.ollama_url, config.llm.ollama_model)
