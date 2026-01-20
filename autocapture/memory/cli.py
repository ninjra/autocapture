"""CLI helpers for the deterministic memory store."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from ..config import AppConfig
from .compiler import ContextCompiler
from .hotness import HotnessPlugin
from .models import ArtifactMeta
from .store import MemoryStore
from .utils import format_utc, stable_json_dumps

_EXIT_USAGE = 2
_EXIT_RETRIEVAL_UNAVAILABLE = 3
_EXIT_VERIFY_FAILED = 4


def add_memory_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="memory_cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest text into the memory store.")
    ingest.add_argument("--path", help="Path to a text file to ingest.")
    ingest.add_argument("--source-uri", help="Optional source URI to record.")
    ingest.add_argument("--title", help="Optional title to record.")
    ingest.add_argument("--label", action="append", default=[], help="Label to attach.")
    ingest.add_argument("--timestamp", help="ISO-8601 timestamp (UTC recommended).")
    ingest.add_argument("--json", action="store_true", help="Emit JSON output.")

    query = sub.add_parser("query", help="Query stored spans via FTS.")
    query.add_argument("query", nargs="?", help="Search query (defaults to stdin).")
    query.add_argument("-k", type=int, default=None, help="Top-K spans to return.")
    query.add_argument("--json", action="store_true", help="Emit JSON output.")

    compile_cmd = sub.add_parser("compile", help="Compile a deterministic context snapshot.")
    compile_cmd.add_argument("query", nargs="?", help="Query (defaults to stdin).")
    compile_cmd.add_argument("-k", type=int, default=None, help="Top-K spans to include.")
    compile_cmd.add_argument("--out", help="Output directory (defaults to snapshots dir).")
    compile_cmd.add_argument("--json", action="store_true", help="Emit JSON output.")

    items = sub.add_parser("items", help="Manage explicit memory items.")
    items_sub = items.add_subparsers(dest="items_cmd", required=True)

    items_list = items_sub.add_parser("list", help="List memory items.")
    items_list.add_argument(
        "--status",
        default="active",
        choices=["active", "proposed", "deprecated", "all"],
        help="Filter by status.",
    )
    items_list.add_argument("--limit", type=int, default=100)
    items_list.add_argument("--offset", type=int, default=0)
    items_list.add_argument("--json", action="store_true", help="Emit JSON output.")

    items_prop = items_sub.add_parser("propose", help="Propose a memory item.")
    items_prop.add_argument("--key", required=True)
    items_prop.add_argument("--value", required=True)
    items_prop.add_argument("--type", dest="item_type", default="fact", help="Item type.")
    items_prop.add_argument("--tag", action="append", default=[])
    items_prop.add_argument("--span-id", action="append", default=[])
    items_prop.add_argument("--user-asserted", action="store_true")
    items_prop.add_argument("--timestamp", help="ISO-8601 timestamp (UTC recommended).")
    items_prop.add_argument("--json", action="store_true", help="Emit JSON output.")

    promote = sub.add_parser("promote", help="Promote a proposed item to active.")
    promote.add_argument("--item-id", required=True)
    promote.add_argument("--span-id", action="append", default=[])
    promote.add_argument("--user-asserted", action="store_true")
    promote.add_argument("--timestamp", help="ISO-8601 timestamp (UTC recommended).")
    promote.add_argument("--json", action="store_true", help="Emit JSON output.")

    verify = sub.add_parser("verify", help="Verify store integrity.")
    verify.add_argument("--json", action="store_true", help="Emit JSON output.")

    gc = sub.add_parser("gc", help="Garbage-collect old snapshots.")
    gc.add_argument("--retention-days", type=int, default=None)
    gc.add_argument("--json", action="store_true", help="Emit JSON output.")

    hotness = sub.add_parser("hotness", help="Memory hotness controls.")
    hot_sub = hotness.add_subparsers(dest="hotness_cmd", required=True)

    hot_top = hot_sub.add_parser("top", help="Rank memory items by hotness.")
    hot_top.add_argument("--as-of", dest="as_of", help="UTC timestamp (YYYY-MM-DDTHH:MM:SSZ).")
    hot_top.add_argument("--now", action="store_true", help="Use current UTC time.")
    hot_top.add_argument("--limit", type=int, default=32)
    hot_top.add_argument("--scope", default="default")
    hot_top.add_argument("--json", action="store_true", help="Emit JSON output.")

    hot_touch = hot_sub.add_parser("touch", help="Record a hotness event for an item.")
    hot_touch.add_argument("--item-id", required=True)
    hot_touch.add_argument("--timestamp", help="UTC timestamp (YYYY-MM-DDTHH:MM:SSZ).")
    hot_touch.add_argument("--now", action="store_true", help="Use current UTC time.")
    hot_touch.add_argument("--signal", default="manual_touch")
    hot_touch.add_argument("--source", default="cli")
    hot_touch.add_argument("--weight", type=float, default=1.0)
    hot_touch.add_argument("--scope", default="default")
    hot_touch.add_argument("--json", action="store_true", help="Emit JSON output.")

    hot_pin = hot_sub.add_parser("pin", help="Pin a memory item.")
    hot_pin.add_argument("--item-id", required=True)
    hot_pin.add_argument("--level", required=True, choices=["hard", "soft"])
    hot_pin.add_argument("--rank", type=int, default=0)
    hot_pin.add_argument("--timestamp", help="UTC timestamp (YYYY-MM-DDTHH:MM:SSZ).")
    hot_pin.add_argument("--now", action="store_true", help="Use current UTC time.")
    hot_pin.add_argument("--source", default="cli")
    hot_pin.add_argument("--scope", default="default")
    hot_pin.add_argument("--json", action="store_true", help="Emit JSON output.")

    hot_unpin = hot_sub.add_parser("unpin", help="Remove a hotness pin.")
    hot_unpin.add_argument("--item-id", required=True)
    hot_unpin.add_argument("--timestamp", help="UTC timestamp (YYYY-MM-DDTHH:MM:SSZ).")
    hot_unpin.add_argument("--now", action="store_true", help="Use current UTC time.")
    hot_unpin.add_argument("--source", default="cli")
    hot_unpin.add_argument("--scope", default="default")
    hot_unpin.add_argument("--json", action="store_true", help="Emit JSON output.")

    hot_gc = hot_sub.add_parser("gc", help="Garbage-collect hotness events.")
    hot_gc.add_argument("--event-days", type=int, default=None)
    hot_gc.add_argument("--event-cap", type=int, default=None)
    hot_gc.add_argument("--as-of", dest="as_of", help="UTC timestamp (YYYY-MM-DDTHH:MM:SSZ).")
    hot_gc.add_argument("--now", action="store_true", help="Use current UTC time.")
    hot_gc.add_argument("--scope", default="default")
    hot_gc.add_argument("--json", action="store_true", help="Emit JSON output.")

    hot_state = hot_sub.add_parser("state", help="Show hotness state for an item.")
    hot_state.add_argument("--item-id", required=True)
    hot_state.add_argument("--scope", default="default")
    hot_state.add_argument("--json", action="store_true", help="Emit JSON output.")


def run_memory_cli(args: argparse.Namespace, config: AppConfig) -> int:
    store = MemoryStore(config.memory)
    hotness = HotnessPlugin(store, config.memory.hotness)

    if args.memory_cmd == "hotness":
        if not config.memory.hotness.enabled:
            _stderr("Memory hotness is disabled in config.")
            return _EXIT_USAGE
        try:
            if args.hotness_cmd == "top":
                as_of = _resolve_timestamp(args.as_of, args.now)
                result = hotness.rank(
                    scope=args.scope,
                    as_of_utc=as_of,
                    limit=args.limit,
                )
                return _emit(result.model_dump(mode="json"), args.json)
            if args.hotness_cmd == "touch":
                ts = _resolve_timestamp(args.timestamp, args.now)
                result = hotness.touch(
                    scope=args.scope,
                    item_id=args.item_id,
                    ts_utc=ts,
                    signal=args.signal,
                    weight=args.weight,
                    source=args.source,
                )
                return _emit(result.model_dump(mode="json"), args.json)
            if args.hotness_cmd == "pin":
                ts = _resolve_timestamp(args.timestamp, args.now)
                result = hotness.pin(
                    scope=args.scope,
                    item_id=args.item_id,
                    level=args.level,
                    rank=args.rank,
                    ts_utc=ts,
                    source=args.source,
                )
                return _emit(result.model_dump(mode="json"), args.json)
            if args.hotness_cmd == "unpin":
                ts = _resolve_timestamp(args.timestamp, args.now)
                result = hotness.unpin(
                    scope=args.scope,
                    item_id=args.item_id,
                    ts_utc=ts,
                    source=args.source,
                )
                return _emit(result.model_dump(mode="json"), args.json)
            if args.hotness_cmd == "gc":
                as_of = _resolve_timestamp(args.as_of, args.now)
                result = hotness.gc(
                    scope=args.scope,
                    as_of_utc=as_of,
                    max_age_days=args.event_days,
                    max_events=args.event_cap,
                )
                return _emit(result.model_dump(mode="json"), args.json)
            if args.hotness_cmd == "state":
                result = hotness.state(scope=args.scope, item_id=args.item_id)
                if result is None:
                    return _emit({"item_id": args.item_id, "found": False}, args.json)
                return _emit(result.model_dump(mode="json"), args.json)
        except ValueError as exc:
            _stderr(str(exc))
            return _EXIT_USAGE

    if args.memory_cmd == "ingest":
        try:
            text, inferred_ts = _load_text(args.path)
        except ValueError as exc:
            _stderr(str(exc))
            return _EXIT_USAGE
        timestamp = args.timestamp or inferred_ts
        meta = ArtifactMeta(
            source_uri=args.source_uri or (args.path or ""),
            title=args.title,
            labels=list(args.label or []),
            content_type="text/plain",
        )
        result = store.ingest_text(text, meta, timestamp=timestamp)
        return _emit(result.model_dump(mode="json"), args.json)

    if args.memory_cmd == "query":
        query = args.query or _read_stdin()
        if not query:
            _stderr("Query required (argument or stdin).")
            return _EXIT_USAGE
        result = store.query_spans(query, k=args.k)
        if result.retrieval_disabled:
            _emit(result.model_dump(mode="json"), args.json)
            return _EXIT_RETRIEVAL_UNAVAILABLE
        return _emit(result.model_dump(mode="json"), args.json)

    if args.memory_cmd == "compile":
        query = args.query or _read_stdin()
        if not query:
            _stderr("Query required (argument or stdin).")
            return _EXIT_USAGE
        compiler = ContextCompiler(store, config.memory)
        out_dir = Path(args.out).expanduser() if args.out else None
        result = compiler.compile(query, k=args.k, output_dir=out_dir)
        return _emit(result.model_dump(mode="json"), args.json)

    if args.memory_cmd == "items":
        if args.items_cmd == "list":
            result = store.list_items(
                status=args.status,
                limit=args.limit,
                offset=args.offset,
            )
            return _emit(result.model_dump(mode="json"), args.json)
        if args.items_cmd == "propose":
            try:
                item = store.propose_item(
                    key=args.key,
                    value=args.value,
                    item_type=args.item_type,
                    tags=args.tag,
                    span_ids=args.span_id,
                    user_asserted=args.user_asserted,
                    timestamp=args.timestamp,
                )
            except ValueError as exc:
                _stderr(str(exc))
                return _EXIT_USAGE
            return _emit(item.model_dump(mode="json"), args.json)

    if args.memory_cmd == "promote":
        try:
            result = store.promote_item(
                item_id=args.item_id,
                span_ids=args.span_id,
                user_asserted=args.user_asserted,
                timestamp=args.timestamp,
            )
        except ValueError as exc:
            _stderr(str(exc))
            return _EXIT_USAGE
        return _emit(result.model_dump(mode="json"), args.json)

    if args.memory_cmd == "verify":
        result = store.verify()
        code = 0 if result.ok else _EXIT_VERIFY_FAILED
        _emit(result.model_dump(mode="json"), args.json)
        return code

    if args.memory_cmd == "gc":
        result = store.gc_snapshots(retention_days=args.retention_days)
        return _emit(result.model_dump(mode="json"), args.json)

    _stderr("Unknown memory command")
    return _EXIT_USAGE


def _emit(payload: dict, as_json: bool) -> int:
    if as_json:
        sys.stdout.write(stable_json_dumps(payload))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(stable_json_dumps(payload))
        sys.stdout.write("\n")
    return 0


def _read_stdin() -> str:
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read().strip()


def _load_text(path_value: str | None) -> tuple[str, str | None]:
    if not path_value:
        text = _read_stdin()
        if not text:
            raise ValueError("Input required (use --path or stdin).")
        return text, None
    path = Path(path_value).expanduser()
    raw = path.read_text(encoding="utf-8")
    ts = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return raw, format_utc(ts)


def _stderr(message: str) -> None:
    sys.stderr.write(message + "\n")


def _resolve_timestamp(value: str | None, use_now: bool) -> str:
    if value:
        return value
    if use_now:
        ts = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        return format_utc(ts)
    raise ValueError("timestamp required (use --timestamp/--as-of or --now)")
