"""Application bootstrap."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from . import configure_logging, load_config
from .capture import CaptureEvent, CaptureService, DirectXDesktopDuplicator
from .config import AppConfig
from .observability import MetricsServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autocapture orchestrator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    return parser.parse_args()


async def main_async(config: AppConfig) -> None:
    metrics = MetricsServer(config.observability)
    metrics.start()

    def handle_capture(event: CaptureEvent) -> None:
        metrics.increment_captures()
        # TODO: enqueue event to OCR queue

    backend = DirectXDesktopDuplicator()
    service = CaptureService(config.capture, backend, handle_capture)
    service.start()

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        service.stop()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_dir, args.log_level)
    config = load_config(args.config)
    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
