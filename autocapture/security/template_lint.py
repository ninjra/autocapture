"""Template linting utilities for prompt safety."""

from __future__ import annotations

import re

from jinja2 import nodes
from jinja2.sandbox import SandboxedEnvironment

_FORBIDDEN_PATTERNS = [
    (re.compile(r"__"), "dunder sequence"),
    (re.compile(r"\|\s*attr\b"), "attr filter"),
    (re.compile(r"map\s*\(\s*attribute\s*="), "map(attribute=...)"),
    (re.compile(r"\|\s*tojson\b"), "tojson filter"),
    (re.compile(r"\|\s*safe\b"), "safe filter"),
]

_BANNED_FILTERS = {
    "attr",
    "map",
    "selectattr",
    "rejectattr",
    "tojson",
    "safe",
}

_BANNED_NAMES = {
    "cycler",
    "joiner",
    "namespace",
}

_FORBIDDEN_NODES = (
    nodes.Import,
    nodes.FromImport,
    nodes.Include,
    nodes.Extends,
    nodes.Macro,
    nodes.CallBlock,
    nodes.Call,
    nodes.Assign,
    nodes.For,
    nodes.If,
    nodes.FilterBlock,
    nodes.Getattr,
    nodes.Getitem,
)


def lint_template_text(template: str, *, label: str = "template") -> None:
    if not template:
        return
    for pattern, reason in _FORBIDDEN_PATTERNS:
        if pattern.search(template):
            raise ValueError(f"{label} contains forbidden pattern: {reason}")
    try:
        parsed = SandboxedEnvironment().parse(template)
    except Exception as exc:
        raise ValueError(f"{label} could not be parsed as a template: {exc}") from exc
    for node_type in _FORBIDDEN_NODES:
        if any(parsed.find_all(node_type)):
            raise ValueError(f"{label} contains forbidden Jinja2 construct: {node_type.__name__}")
    for node in parsed.find_all(nodes.Filter):
        if node.name in _BANNED_FILTERS:
            raise ValueError(f"{label} contains forbidden filter: {node.name}")
    for node in parsed.find_all(nodes.Name):
        if node.name in _BANNED_NAMES:
            raise ValueError(f"{label} references forbidden global: {node.name}")
