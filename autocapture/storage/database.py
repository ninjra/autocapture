"""Database session management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import DatabaseConfig
from ..logging_utils import get_logger


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
        if config.url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            engine_kwargs["pool_size"] = config.pool_size
            engine_kwargs["max_overflow"] = config.max_overflow
        self._engine = create_engine(config.url, **engine_kwargs)
        self._run_migrations()
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._log.info("Database connected at %s", config.url)

    @property
    def engine(self):
        return self._engine

    def _run_migrations(self) -> None:
        import importlib.util

        if importlib.util.find_spec("alembic.command") is None:  # pragma: no cover
            from .models import Base

            self._log.warning("Alembic not available; falling back to metadata create_all")
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
