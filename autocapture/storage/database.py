"""Database session management."""

from __future__ import annotations

from contextlib import contextmanager
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterator, TypeVar

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import DatabaseConfig
from ..logging_utils import get_logger
from ..paths import resource_root
from ..security.sqlcipher import load_sqlcipher_key

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
        self._sqlcipher_key: bytes | None = None
        self._ran_migrations = False
        engine_kwargs = {
            "echo": config.echo,
            "pool_pre_ping": True,
            "pool_recycle": 1800,
        }
        is_sqlite = config.url.startswith("sqlite")
        is_memory = self._is_sqlite_memory(config.url)
        self._is_memory = is_memory
        if is_sqlite and not is_memory:
            self._enforce_secure_mode()
        if is_sqlite and config.encryption_enabled and not is_memory:
            self._sqlcipher_key = self._load_sqlcipher_key()
            engine_kwargs["module"] = self._load_sqlcipher_module()
        if is_sqlite:
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": config.sqlite_busy_timeout_ms / 1000,
            }
            if is_memory:
                engine_kwargs["poolclass"] = StaticPool
        else:
            engine_kwargs["pool_size"] = config.pool_size
            engine_kwargs["max_overflow"] = config.max_overflow
        self._engine = create_engine(config.url, **engine_kwargs)
        if is_sqlite:
            self._register_sqlite_pragmas(is_memory)
        if is_memory:
            init_schema(self._engine)
            self._ran_migrations = True
        else:
            self._run_migrations()
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._log.info("Database connected at {}", config.url)

    def _enforce_secure_mode(self) -> None:
        if not self._config.secure_mode_required:
            return
        if self._config.encryption_enabled:
            return
        if self._config.allow_insecure_dev or os.environ.get("AUTOCAPTURE_ALLOW_INSECURE_DEV"):
            self._log.warning(
                "Secure mode required but allow_insecure_dev is set; running without SQLCipher."
            )
            return
        raise RuntimeError(
            "Secure mode required: enable database.encryption_enabled or set "
            "database.allow_insecure_dev=true for development."
        )

    @property
    def engine(self):
        return self._engine

    def _run_migrations(self) -> None:
        import importlib.util

        if importlib.util.find_spec("alembic.command") is None:  # pragma: no cover
            from .models import Base

            self._log.warning("Alembic not available; falling back to metadata create_all")
            Base.metadata.create_all(self._engine)
            self._ran_migrations = True
            return

        from alembic import command  # type: ignore
        from alembic.config import Config  # type: ignore

        base_dir = resource_root()
        config_path = base_dir / "alembic.ini"
        script_location = base_dir / "alembic"
        if not config_path.exists() or not script_location.exists():
            from .models import Base

            self._log.warning("Alembic config not found; falling back to metadata create_all")
            Base.metadata.create_all(self._engine)
            self._ran_migrations = True
            return
        alembic_cfg = Config(str(config_path))
        alembic_cfg.set_main_option("sqlalchemy.url", self._config.url)
        alembic_cfg.set_main_option("script_location", str(script_location))
        command.upgrade(alembic_cfg, "head")
        self._ran_migrations = True

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
        key = self._sqlcipher_key

        @event.listens_for(self._engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                if key:
                    cursor.execute("PRAGMA key = ?", (key,))
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute(f"PRAGMA busy_timeout={config.sqlite_busy_timeout_ms}")
                if not is_memory and config.sqlite_wal:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute(f"PRAGMA synchronous={config.sqlite_synchronous}")
                elif not is_memory:
                    cursor.execute(f"PRAGMA synchronous={config.sqlite_synchronous}")
            finally:
                cursor.close()

    def _load_sqlcipher_module(self):
        import importlib.util

        if importlib.util.find_spec("pysqlcipher3") is None:
            raise RuntimeError(
                "SQLCipher support requires pysqlcipher3. Install via: "
                "poetry install --extras sqlcipher (Windows uses rotki-pysqlcipher3 wheels)."
            )
        import pysqlcipher3.dbapi2 as sqlcipher  # type: ignore

        return sqlcipher

    def _load_sqlcipher_key(self) -> bytes:
        data_dir = Path(self._config.url.replace("sqlite:///", "")).parent
        return load_sqlcipher_key(self._config, data_dir)

    @staticmethod
    def _is_sqlite_memory(url: str) -> bool:
        return ":memory:" in url or "mode=memory" in url

    @staticmethod
    def _is_sqlite_lock(exc: OperationalError) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database is busy" in message
