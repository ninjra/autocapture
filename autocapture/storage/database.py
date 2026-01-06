"""Database session management."""

from __future__ import annotations

from contextlib import contextmanager
import random
import time
from typing import Callable, Iterator, TypeVar

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from ..config import DatabaseConfig
from ..logging_utils import get_logger

T = TypeVar("T")


def init_schema(engine) -> None:
    """Create all SQLAlchemy tables. Safe to call multiple times."""
    log = get_logger("db")
    try:
        from . import models as storage_models

        storage_models.Base.metadata.create_all(bind=engine)
        log.info("DB schema initialized via Base.metadata.create_all().")
    except Exception:
        log.exception("DB schema initialization failed.")
        raise


class DatabaseManager:
    """Configure SQLAlchemy engine and provide sessions."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._log = get_logger("db")
        engine_kwargs = {
            "echo": config.echo,
            "pool_pre_ping": True,
            "pool_recycle": 1800,
        }
        is_sqlite = config.url.startswith("sqlite")
        is_memory = self._is_sqlite_memory(config.url)
        if is_sqlite:
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": config.sqlite_busy_timeout_ms / 1000,
            }
        else:
            engine_kwargs["pool_size"] = config.pool_size
            engine_kwargs["max_overflow"] = config.max_overflow
        self._engine = create_engine(config.url, **engine_kwargs)
        if is_sqlite:
            self._register_sqlite_pragmas(is_memory)
        self._run_migrations()
        if self._engine.dialect.name == "sqlite" and (
            ":memory:" in str(self._engine.url)
            or "mode=memory" in str(self._engine.url)
        ):
            init_schema(self._engine)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._log.info("Database connected at {}", config.url)

    @property
    def engine(self):
        return self._engine

    def _run_migrations(self) -> None:
        import importlib.util

        if importlib.util.find_spec("alembic.command") is None:  # pragma: no cover
            from .models import Base

            self._log.warning(
                "Alembic not available; falling back to metadata create_all"
            )
            Base.metadata.create_all(self._engine)
            return

        from alembic import command  # type: ignore
        from alembic.config import Config  # type: ignore

        config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
        alembic_cfg = Config(str(config_path))
        alembic_cfg.set_main_option("sqlalchemy.url", self._config.url)
        alembic_cfg.set_main_option(
            "script_location",
            str(Path(__file__).resolve().parents[2] / "alembic"),
        )
        command.upgrade(alembic_cfg, "head")

    @contextmanager
    def session(self) -> Iterator[Session]:
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def transaction(
        self,
        func: Callable[[Session], T],
        *,
        max_retries: int = 6,
    ) -> T:
        attempt = 0
        while True:
            session: Session = self._session_factory()
            try:
                result = func(session)
                session.commit()
                return result
            except OperationalError as exc:
                session.rollback()
                if self._is_sqlite_lock(exc) and attempt < max_retries:
                    delay = min(0.025 * (2**attempt), 0.2)
                    delay += random.uniform(0, 0.05)
                    if attempt == 0:
                        self._log.warning("SQLite busy; retrying transaction")
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def _register_sqlite_pragmas(self, is_memory: bool) -> None:
        config = self._config

        @event.listens_for(self._engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute(f"PRAGMA busy_timeout={config.sqlite_busy_timeout_ms}")
                if not is_memory and config.sqlite_wal:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute(f"PRAGMA synchronous={config.sqlite_synchronous}")
                elif not is_memory:
                    cursor.execute(f"PRAGMA synchronous={config.sqlite_synchronous}")
            finally:
                cursor.close()

    @staticmethod
    def _is_sqlite_memory(url: str) -> bool:
        return ":memory:" in url or "mode=memory" in url

    @staticmethod
    def _is_sqlite_lock(exc: OperationalError) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database is busy" in message
