"""TRON (Token Reduced Object Notation) encoder/decoder."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_CLASS_DEF_RE = re.compile(r"^\\s*class\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\{([^}]*)\\}\\s*$")

_PREFERRED_FIELD_ORDER = {
    "ContextPack": ["version", "query", "generated_at", "evidence", "warnings"],
    "EvidenceItem": ["id", "ts_start", "ts_end", "source", "title", "text", "meta"],
    "EvidenceSpan": ["span_id", "start", "end", "conf"],
}

_LIST_CLASS_BY_KEY = {
    "evidence": "EvidenceItem",
    "spans": "EvidenceSpan",
}


def encode_tron(value: Any) -> str:
    encoder = _TronEncoder()
    return encoder.encode(value)


def decode_tron(text: str) -> Any:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty TRON payload")
    try:
        return json.loads(raw)
    except Exception:
        pass
    class_defs, body = _split_class_defs(raw)
    parser = _TronParser(body, class_defs)
    return parser.parse()


def _split_class_defs(text: str) -> tuple[dict[str, list[str]], str]:
    class_defs: dict[str, list[str]] = {}
    body_lines: list[str] = []
    in_header = True
    for line in text.splitlines():
        if in_header:
            match = _CLASS_DEF_RE.match(line)
            if match:
                name = match.group(1)
                fields = [field for field in match.group(2).split() if field]
                class_defs[name] = fields
                continue
            if not line.strip():
                continue
            in_header = False
        body_lines.append(line)
    return class_defs, "\n".join(body_lines).strip()


class _TronEncoder:
    def __init__(self) -> None:
        self._class_defs: dict[str, list[str]] = {}

    def encode(self, value: Any) -> str:
        body = self._encode_value(value, parent_key=None)
        if not self._class_defs:
            return body
        header = "\n".join(self._format_class_def(name) for name in sorted(self._class_defs))
        return f"{header}\n\n{body}"

    def _format_class_def(self, name: str) -> str:
        fields = self._class_defs[name]
        field_list = " ".join(fields)
        return f"class {name} {{ {field_list} }}"

    def _encode_value(self, value: Any, parent_key: str | None) -> str:
        if isinstance(value, dict):
            return self._encode_object(value)
        if isinstance(value, list):
            return self._encode_list(value, parent_key)
        return _encode_scalar(value)

    def _encode_object(self, value: dict) -> str:
        keys = _ordered_keys(value)
        parts = []
        for key in keys:
            encoded = self._encode_value(value[key], parent_key=key)
            parts.append(f"{json.dumps(key)}: {encoded}")
        return "{ " + ", ".join(parts) + " }"

    def _encode_list(self, value: list, parent_key: str | None) -> str:
        if parent_key in _LIST_CLASS_BY_KEY and _is_uniform_dict_list(value):
            class_name = _LIST_CLASS_BY_KEY[parent_key]
            fields = _ordered_keys(value[0], class_name=class_name)
            self._class_defs.setdefault(class_name, fields)
            items = []
            for item in value:
                args = [self._encode_value(item[field], parent_key=None) for field in fields]
                items.append(f"{class_name}(" + ", ".join(args) + ")")
            return "[ " + ", ".join(items) + " ]"
        items = [self._encode_value(item, parent_key=None) for item in value]
        return "[ " + ", ".join(items) + " ]"


def _ordered_keys(value: dict, class_name: str | None = None) -> list[str]:
    if class_name in _PREFERRED_FIELD_ORDER:
        ordered = [key for key in _PREFERRED_FIELD_ORDER[class_name] if key in value]
        rest = sorted(key for key in value if key not in ordered)
        return ordered + rest
    if _looks_like_context_pack(value):
        ordered = [key for key in _PREFERRED_FIELD_ORDER["ContextPack"] if key in value]
        rest = sorted(key for key in value if key not in ordered)
        return ordered + rest
    return sorted(value.keys())


def _looks_like_context_pack(value: dict) -> bool:
    return {"version", "query", "generated_at", "evidence", "warnings"}.issubset(value.keys())


def _is_uniform_dict_list(value: list) -> bool:
    if not value or not all(isinstance(item, dict) for item in value):
        return False
    first_keys = set(value[0].keys())
    return all(set(item.keys()) == first_keys for item in value)


def _encode_scalar(value: Any) -> str:
    if isinstance(value, (str, bool)) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(str(value))


@dataclass
class _Token:
    kind: str
    value: Any


class _Tokenizer:
    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0

    def peek(self) -> _Token | None:
        pos = self._pos
        token = self.next_token()
        self._pos = pos
        return token

    def next_token(self) -> _Token | None:
        self._skip_ws()
        if self._pos >= len(self._text):
            return None
        ch = self._text[self._pos]
        if ch in "{}[](),:":
            self._pos += 1
            return _Token("punct", ch)
        if ch == '"':
            return _Token("string", self._read_string())
        if ch.isdigit() or ch == "-":
            return _Token("number", self._read_number())
        if ch.isalpha() or ch in "_$":
            return _Token("ident", self._read_ident())
        raise ValueError(f"Unexpected character: {ch}")

    def _skip_ws(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _read_string(self) -> str:
        start = self._pos
        self._pos += 1
        escaped = False
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch == '"' and not escaped:
                self._pos += 1
                break
            escaped = ch == "\\" and not escaped
            self._pos += 1
        return json.loads(self._text[start:self._pos])

    def _read_number(self) -> Any:
        start = self._pos
        while self._pos < len(self._text) and self._text[self._pos] in "0123456789+-.eE":
            self._pos += 1
        return json.loads(self._text[start:self._pos])

    def _read_ident(self) -> str:
        start = self._pos
        while self._pos < len(self._text) and self._text[self._pos] in (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        ):
            self._pos += 1
        return self._text[start:self._pos]


class _TronParser:
    def __init__(self, text: str, class_defs: dict[str, list[str]]) -> None:
        self._tokenizer = _Tokenizer(text)
        self._class_defs = class_defs

    def parse(self) -> Any:
        value = self._parse_value()
        return value

    def _parse_value(self) -> Any:
        token = self._tokenizer.next_token()
        if token is None:
            raise ValueError("Unexpected end of input")
        if token.kind == "string":
            return token.value
        if token.kind == "number":
            return token.value
        if token.kind == "ident":
            if token.value in {"true", "false", "null"}:
                return {"true": True, "false": False, "null": None}[token.value]
            peek = self._tokenizer.peek()
            if peek and peek.kind == "punct" and peek.value == "(":
                return self._parse_class_instance(token.value)
            return token.value
        if token.kind == "punct" and token.value == "{":
            return self._parse_object()
        if token.kind == "punct" and token.value == "[":
            return self._parse_array()
        raise ValueError(f"Unexpected token: {token}")

    def _parse_object(self) -> dict:
        obj: dict[str, Any] = {}
        while True:
            token = self._tokenizer.peek()
            if token is None:
                raise ValueError("Unterminated object")
            if token.kind == "punct" and token.value == "}":
                self._tokenizer.next_token()
                return obj
            key_token = self._tokenizer.next_token()
            if key_token is None or key_token.kind not in {"string", "ident"}:
                raise ValueError("Invalid object key")
            key = str(key_token.value)
            colon = self._tokenizer.next_token()
            if colon is None or colon.kind != "punct" or colon.value != ":":
                raise ValueError("Expected ':' after key")
            obj[key] = self._parse_value()
            token = self._tokenizer.peek()
            if token and token.kind == "punct" and token.value == ",":
                self._tokenizer.next_token()
                continue

    def _parse_array(self) -> list:
        items: list[Any] = []
        while True:
            token = self._tokenizer.peek()
            if token is None:
                raise ValueError("Unterminated array")
            if token.kind == "punct" and token.value == "]":
                self._tokenizer.next_token()
                return items
            items.append(self._parse_value())
            token = self._tokenizer.peek()
            if token and token.kind == "punct" and token.value == ",":
                self._tokenizer.next_token()
                continue

    def _parse_class_instance(self, name: str) -> dict:
        open_paren = self._tokenizer.next_token()
        if not open_paren or open_paren.kind != "punct" or open_paren.value != "(":
            raise ValueError("Expected '(' after class name")
        args: list[Any] = []
        while True:
            token = self._tokenizer.peek()
            if token is None:
                raise ValueError("Unterminated class instance")
            if token.kind == "punct" and token.value == ")":
                self._tokenizer.next_token()
                break
            args.append(self._parse_value())
            token = self._tokenizer.peek()
            if token and token.kind == "punct" and token.value == ",":
                self._tokenizer.next_token()
                continue
        fields = self._class_defs.get(name)
        if not fields:
            raise ValueError(f"Unknown class: {name}")
        data: dict[str, Any] = {}
        for idx, field in enumerate(fields):
            data[field] = args[idx] if idx < len(args) else None
        return data
