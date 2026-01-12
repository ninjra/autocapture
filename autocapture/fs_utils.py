"""Filesystem helpers for atomic and cross-platform safe writes."""

from __future__ import annotations

import errno
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


def fsync_file(path: Path) -> None:
    # On Windows, os.fsync on a read-only handle can raise EBADF.
    # Use a writable handle when possible; fall back gracefully for edge cases.
    mode = "r+b" if os.name == "nt" else "rb"
    try:
        with path.open(mode) as handle:
            try:
                handle.flush()
            except Exception:
                # flush() may not be meaningful for read handles; ignore.
                pass
            os.fsync(handle.fileno())
    except FileNotFoundError:
        # File disappeared between operations; caller will handle missing files later.
        return
    except OSError as exc:
        # Windows can still raise for some filesystem/handle types. Best-effort fsync.
        if os.name == "nt" and getattr(exc, "errno", None) in (9, 22, 13):
            return
        raise


def fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_unlink(path: Path, retries: int = 5, backoff_s: float = 0.05) -> None:
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s * (2**attempt))


def safe_replace(source: Path, destination: Path) -> None:
    try:
        os.replace(source, destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    _copy_atomic(source, destination)
    safe_unlink(source)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".tmp-{destination.name}-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as src, temp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        fsync_dir(temp_path.parent)
        os.replace(temp_path, destination)
        fsync_dir(destination.parent)
    finally:
        if temp_path.exists():
            safe_unlink(temp_path)


def atomic_publish(
    source: Path,
    destination: Path,
    writer: Callable[[Path, Path], None],
    remove_source: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".tmp-{destination.name}-{uuid.uuid4().hex}")
    try:
        writer(source, temp_path)
        fsync_file(temp_path)
        fsync_dir(temp_path.parent)
        safe_replace(temp_path, destination)
        fsync_dir(destination.parent)
    finally:
        if temp_path.exists():
            safe_unlink(temp_path)
    if remove_source:
        safe_unlink(source)


@contextmanager
def file_lock(path: Path, timeout_s: float = 10.0, poll_interval_s: float = 0.05) -> Iterator[None]:
    start = time.monotonic()
    fd = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if time.monotonic() - start >= timeout_s:
                raise TimeoutError(f"Timed out waiting for lock {path}")
            time.sleep(poll_interval_s)
    try:
        yield None
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
