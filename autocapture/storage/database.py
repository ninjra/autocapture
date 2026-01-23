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


def _ensure_create_function_compat(dbapi_connection) -> None:
    """Patch DBAPIs that don't accept the 'deterministic' kwarg (pysqlcipher3)."""

    create_fn = getattr(dbapi_connection, "create_function", None)
    if create_fn is None:
        return
    try:
        create_fn("_autocapture_probe", 1, lambda x: x, deterministic=True)
        return
    except TypeError as exc:
        if "at most 3 arguments" not in str(exc):
            return
    except Exception:
        return

    # pysqlcipher3 lacks deterministic kwarg; patch instance first, then class.
    def _compat(name, num_params, func, deterministic=None):  # noqa: ANN001
        _ = deterministic
        return create_fn(name, num_params, func)

    try:
        setattr(dbapi_connection, "create_function", _compat)
        return
    except Exception:
        pass
    try:
        cls = type(dbapi_connection)
        original = cls.create_function
    except Exception:
        return

    def _compat_class(self, name, num_params, func, deterministic=None):  # noqa: ANN001
        _ = deterministic
        return original(self, name, num_params, func)

    try:
        setattr(cls, "create_function", _compat_class)
    except Exception:
        return


def _apply_sqlcipher_key(cursor, key: bytes) -> None:
    """Apply SQLCipher key using param binding, falling back to hex literals if needed."""

    hex_key = key.hex()
    try:
        # Use hex literal to avoid truncation issues with NULL bytes in raw keys.
        cursor.execute(f"PRAGMA key = \"x'{hex_key}'\"")
        return
    except Exception:
        pass
    try:
        cursor.execute("PRAGMA key = ?", (key,))
    except Exception as exc:
        message = str(exc).lower()
        if 'near "?"' not in message and "near '?'" not in message:
            raise
        cursor.execute(f"PRAGMA key = \"x'{hex_key}'\"")


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
        self._sqlcipher_module = None
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
        connect_args = {}
        if is_sqlite:
            connect_args = {"check_same_thread": False}
            if not is_memory:
                connect_args["timeout"] = config.sqlite_busy_timeout_ms / 1000
        if is_sqlite and config.encryption_enabled and not is_memory:
            self._sqlcipher_key = self._load_sqlcipher_key()
            self._sqlcipher_module = self._load_sqlcipher_module()
            engine_kwargs["creator"] = self._make_sqlcipher_creator(connect_args)
            engine_kwargs["module"] = self._sqlcipher_module
        elif is_sqlite:
            engine_kwargs["connect_args"] = connect_args
            if is_memory:
                engine_kwargs["poolclass"] = StaticPool
        else:
            engine_kwargs["pool_size"] = config.pool_size
            engine_kwargs["max_overflow"] = config.max_overflow
        self._engine = create_engine(config.url, **engine_kwargs)
        if is_sqlite and config.encryption_enabled and not is_memory:
            self._apply_sqlcipher_dialect_compat()
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
        if self._config.encryption_enabled and self._config.url.startswith("sqlite"):
            with self._engine.connect() as connection:
                alembic_cfg.attributes["connection"] = connection
                command.upgrade(alembic_cfg, "head")
        else:
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

        @event.listens_for(self._engine, "connect", insert=True)
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            _ensure_create_function_compat(dbapi_connection)
            cursor = dbapi_connection.cursor()
            try:
                if key:
                    _apply_sqlcipher_key(cursor, key)
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
        from ..security.sqlcipher import ensure_sqlcipher_create_function_compat

        ensure_sqlcipher_create_function_compat(sqlcipher)

        return sqlcipher

    def _load_sqlcipher_key(self) -> bytes:
        data_dir = Path(self._config.url.replace("sqlite:///", "")).parent
        return load_sqlcipher_key(self._config, data_dir)

    def _apply_sqlcipher_dialect_compat(self) -> None:
        """Force SQLAlchemy to avoid deterministic kwarg for SQLCipher DBAPI."""

        dialect = getattr(self._engine, "dialect", None)
        if dialect is None:
            return
        try:
            setattr(dialect, "_sqlite_version_info", (3, 8, 2))
        except Exception:
            pass
        try:
            setattr(dialect, "_deterministic", False)
        except Exception:
            pass
        if self._sqlcipher_module is not None:
            try:
                setattr(dialect, "dbapi", self._sqlcipher_module)
            except Exception:
                return

    def _sqlite_path_from_url(self, url: str) -> str:
        if url.startswith("sqlite:////"):
            return url.replace("sqlite:////", "/")
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "")
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "")
        return url

    def _make_sqlcipher_creator(self, connect_args: dict) -> Callable[[], object]:
        sqlcipher = self._sqlcipher_module
        key = self._sqlcipher_key
        db_path = self._sqlite_path_from_url(self._config.url)

        def _creator():
            conn = sqlcipher.connect(db_path, **connect_args)  # type: ignore[call-arg]
            _ensure_create_function_compat(conn)
            if key:
                cursor = conn.cursor()
                try:
                    _apply_sqlcipher_key(cursor, key)
                finally:
                    cursor.close()
            return conn

        return _creator

    @staticmethod
    def _is_sqlite_memory(url: str) -> bool:
        return ":memory:" in url or "mode=memory" in url

    @staticmethod
    def _is_sqlite_lock(exc: OperationalError) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database is busy" in message
