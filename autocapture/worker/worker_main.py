"""Background worker loop for OCR, embeddings, and retention."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import AppConfig
from ..logging_utils import get_logger


@dataclass(slots=True)
class WorkerJob:
    job_id: int
    job_type: str
    payload: dict[str, Any]
    attempts: int


class WorkerDatabase:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self._conn.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS worker_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_run_at TEXT,
                lease_expires_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                processed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                segment_id TEXT NOT NULL,
                roi_path TEXT,
                ocr_text TEXT,
                vision_summary TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS segment_embeddings (
                segment_id TEXT PRIMARY KEY,
                hnsw_label INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts USING fts5(
                segment_id,
                content
            );
            """
        )
        self._conn.commit()

    def lease_next_job(self, lease_ms: int) -> WorkerJob | None:
        now = datetime.now(timezone.utc)
        lease_until = now + timedelta(milliseconds=lease_ms)
        now_str = now.isoformat()
        lease_str = lease_until.isoformat()
        cursor = self._conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        row = cursor.execute(
            """
            SELECT id, job_type, payload, attempts
            FROM worker_jobs
            WHERE status IN ('pending', 'retry')
              AND (next_run_at IS NULL OR next_run_at <= ?)
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (now_str, now_str),
        ).fetchone()
        if row is None:
            self._conn.commit()
            return None
        cursor.execute(
            """
            UPDATE worker_jobs
            SET status = 'leased', lease_expires_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (lease_str, now_str, row["id"]),
        )
        self._conn.commit()
        return WorkerJob(
            job_id=row["id"],
            job_type=row["job_type"],
            payload=json.loads(row["payload"]),
            attempts=row["attempts"],
        )

    def mark_done(self, job_id: int) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            UPDATE worker_jobs
            SET status = 'done', lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (now_str, job_id),
        )
        self._conn.commit()

    def mark_retry(self, job_id: int, attempts: int, error: str) -> None:
        now = datetime.now(timezone.utc)
        delay_s = min(900, 2**min(attempts, 10))
        next_run = now + timedelta(seconds=delay_s)
        self._conn.execute(
            """
            UPDATE worker_jobs
            SET status = 'retry',
                attempts = ?,
                next_run_at = ?,
                lease_expires_at = NULL,
                last_error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                attempts,
                next_run.isoformat(),
                error,
                now.isoformat(),
                job_id,
            ),
        )
        self._conn.commit()

    def backlog_for(self, job_type: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(1) AS count
            FROM worker_jobs
            WHERE status IN ('pending', 'retry') AND job_type = ?
            """,
            (job_type,),
        ).fetchone()
        return int(row["count"]) if row else 0

    def fetch_observation(self, observation_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM observations WHERE id = ?
            """,
            (observation_id,),
        ).fetchone()

    def update_observation_text(
        self, observation_id: str, ocr_text: str, updated_at: str
    ) -> None:
        self._conn.execute(
            """
            UPDATE observations
            SET ocr_text = ?, updated_at = ?
            WHERE id = ?
            """,
            (ocr_text, updated_at, observation_id),
        )
        self._conn.commit()

    def fetch_segment_text(self, segment_id: str) -> str:
        rows = self._conn.execute(
            """
            SELECT ocr_text, vision_summary
            FROM observations
            WHERE segment_id = ?
            """,
            (segment_id,),
        ).fetchall()
        chunks: list[str] = []
        for row in rows:
            for field in (row["ocr_text"], row["vision_summary"]):
                if field:
                    chunks.append(str(field))
        return " ".join(chunks).strip()

    def upsert_segment_fts(self, segment_id: str, content: str) -> None:
        self._conn.execute(
            "DELETE FROM segment_fts WHERE segment_id = ?",
            (segment_id,),
        )
        self._conn.execute(
            "INSERT INTO segment_fts(segment_id, content) VALUES(?, ?)",
            (segment_id, content),
        )
        self._conn.commit()

    def mark_segment_processed(self, segment_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            UPDATE segments
            SET state = 'processed', processed_at = ?
            WHERE id = ?
            """,
            (now, segment_id),
        )
        self._conn.commit()

    def get_embedding_label(self, segment_id: str) -> int | None:
        row = self._conn.execute(
            """
            SELECT hnsw_label FROM segment_embeddings WHERE segment_id = ?
            """,
            (segment_id,),
        ).fetchone()
        return int(row["hnsw_label"]) if row else None

    def next_embedding_label(self) -> int:
        row = self._conn.execute(
            """
            SELECT MAX(hnsw_label) AS max_label FROM segment_embeddings
            """,
        ).fetchone()
        return int(row["max_label"] or -1) + 1

    def store_embedding_label(self, segment_id: str, label: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO segment_embeddings(segment_id, hnsw_label, updated_at)
            VALUES(?, ?, ?)
            """,
            (segment_id, label, now),
        )
        self._conn.commit()

    def observations_for_retention(self, cutoff: datetime) -> Iterable[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT observations.roi_path AS roi_path
            FROM observations
            JOIN segments ON segments.id = observations.segment_id
            WHERE segments.state = 'processed'
              AND observations.created_at <= ?
              AND observations.roi_path IS NOT NULL
            """,
            (cutoff.isoformat(),),
        ).fetchall()


class OCRProcessor:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self._warmup()

    def _warmup(self) -> None:
        sample = np.zeros((16, 16, 3), dtype=np.uint8)
        self._engine(sample)

    def run(self, image: np.ndarray) -> list[tuple[str, float, list[int]]]:
        results, _ = self._engine(image)
        spans = []
        for result in results or []:
            box, text, confidence = result
            flattened = [int(coord) for point in box for coord in point]
            spans.append((text, float(confidence), flattened))
        return spans


class EmbeddingIndex:
    def __init__(self, model_name: str, index_path: Path) -> None:
        from fastembed import TextEmbedding
        import hnswlib

        self._log = get_logger("embeddings")
        self._model = TextEmbedding(model_name)
        self._index_path = index_path
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = hnswlib.Index(space="cosine", dim=self._embedding_dim())
        if self._index_path.exists():
            self._index.load_index(str(self._index_path))
        else:
            self._index.init_index(max_elements=1024, ef_construction=200, M=16)
        self._index.set_ef(128)

    def _embedding_dim(self) -> int:
        sample = next(self._model.embed(["warmup"]))
        return int(len(sample))

    def ensure_capacity(self, extra: int = 1) -> None:
        current = self._index.get_current_count()
        max_elements = self._index.get_max_elements()
        if current + extra <= max_elements:
            return
        new_max = max(max_elements * 2, current + extra)
        self._log.info("Resizing HNSW index to {}", new_max)
        self._index.resize_index(new_max)

    def add_embedding(self, label: int, text: str) -> None:
        vector = next(self._model.embed([text]))
        self.ensure_capacity()
        self._index.add_items(np.array([vector]), np.array([label]))
        self._index.save_index(str(self._index_path))


class RetentionManager:
    def __init__(self, data_dir: Path, policy: Any) -> None:
        self._data_dir = data_dir
        self._policy = policy
        self._log = get_logger("retention")

    def enforce(self, db: WorkerDatabase) -> None:
        self._delete_old_videos()
        self._delete_old_roi_images(db)
        self._enforce_max_media()

    def _delete_old_videos(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._policy.video_days)
        video_dir = self._data_dir / "video"
        if not video_dir.exists():
            return
        removed = 0
        for path in sorted(video_dir.rglob("*")):
            if not path.is_file():
                continue
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if timestamp <= cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        if removed:
            self._log.info("Removed {} old video files", removed)

    def _delete_old_roi_images(self, db: WorkerDatabase) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._policy.roi_days)
        removed = 0
        for row in db.observations_for_retention(cutoff):
            roi_path = Path(row["roi_path"])
            if roi_path.exists():
                roi_path.unlink(missing_ok=True)
                removed += 1
        if removed:
            self._log.info("Removed {} old ROI images", removed)

    def _enforce_max_media(self) -> None:
        max_bytes = int(self._policy.max_media_gb * (1024**3))
        video_dir = self._data_dir / "video"
        if not video_dir.exists():
            return
        total = sum(
            path.stat().st_size
            for path in video_dir.rglob("*")
            if path.is_file()
        )
        if total <= max_bytes:
            return
        files = [
            path
            for path in video_dir.rglob("*")
            if path.is_file()
        ]
        files.sort(key=lambda path: path.stat().st_mtime)
        removed = 0
        for path in files:
            if total <= max_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
            removed += 1
        if removed:
            self._log.warning("Removed {} videos to respect media cap", removed)


class Worker:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._log = get_logger("worker")
        self._data_dir = config.worker.data_dir
        self._db = WorkerDatabase(self._data_dir / "db" / "autocapture.sqlite")
        self._ocr = OCRProcessor()
        self._embeddings = EmbeddingIndex(
            config.embeddings.model,
            self._data_dir / "index" / "segments_hnsw.bin",
        )
        self._retention = RetentionManager(self._data_dir, config.retention)

    def run(self) -> None:
        lease_ms = self._config.worker.lease_ms
        poll_interval = self._config.worker.poll_interval_s
        self._log.info("Worker loop starting")
        while True:
            job = self._db.lease_next_job(lease_ms)
            if job is None:
                time.sleep(poll_interval)
                continue
            start = time.perf_counter()
            try:
                self._process_job(job)
                self._db.mark_done(job.job_id)
                status = "done"
            except Exception as exc:
                attempts = job.attempts + 1
                self._db.mark_retry(job.job_id, attempts, str(exc))
                status = "retry"
                self._log.bind(job_id=job.job_id, job_type=job.job_type).exception(
                    "Job failed"
                )
            duration_ms = (time.perf_counter() - start) * 1000
            self._log.bind(
                job_id=job.job_id,
                job_type=job.job_type,
                status=status,
                duration_ms=round(duration_ms, 2),
            ).info("Job completed")

    def _process_job(self, job: WorkerJob) -> None:
        if job.job_type == "ocr_observation":
            self._handle_ocr(job.payload)
            return
        if job.job_type == "segment_finalize":
            self._handle_segment_finalize(job.payload)
            return
        if job.job_type == "retention":
            self._retention.enforce(self._db)
            return
        raise ValueError(f"Unknown job type: {job.job_type}")

    def _handle_ocr(self, payload: dict[str, Any]) -> None:
        observation_id = payload["observation_id"]
        observation = self._db.fetch_observation(observation_id)
        if observation is None:
            raise ValueError(f"Observation not found: {observation_id}")
        roi_path = Path(observation["roi_path"])
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI image missing: {roi_path}")
        backlog = self._db.backlog_for("ocr_observation")
        if backlog > self._config.worker.ocr_backlog_soft_limit:
            delay = min(1.0, backlog / 5000)
            time.sleep(delay)
        from PIL import Image

        image = np.array(Image.open(roi_path).convert("RGB"))
        spans = self._ocr.run(image)
        text = " ".join(span[0] for span in spans if span[0])
        normalized = " ".join(text.split())
        now_str = datetime.now(timezone.utc).isoformat()
        self._db.update_observation_text(observation_id, normalized, now_str)

    def _handle_segment_finalize(self, payload: dict[str, Any]) -> None:
        segment_id = payload["segment_id"]
        text = self._db.fetch_segment_text(segment_id)
        if text:
            self._db.upsert_segment_fts(segment_id, text)
            label = self._db.get_embedding_label(segment_id)
            if label is None:
                label = self._db.next_embedding_label()
                self._embeddings.add_embedding(label, text)
                self._db.store_embedding_label(segment_id, label)
        self._db.mark_segment_processed(segment_id)


def main(config: AppConfig) -> None:
    worker = Worker(config)
    worker.run()


__all__ = ["main", "Worker"]
