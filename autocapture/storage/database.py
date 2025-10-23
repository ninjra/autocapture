"""Database session management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import DatabaseConfig
from ..logging_utils import get_logger
from .models import Base


class DatabaseManager:
    """Configure SQLAlchemy engine and provide sessions."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._log = get_logger("db")
        self._engine = create_engine(
            config.url,
            echo=config.echo,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._log.info("Database connected at %s", config.url)

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
