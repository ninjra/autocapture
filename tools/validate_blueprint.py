"""Validate the BLUEPRINT.md artifact for structure and coverage consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_HEADINGS = [
    "# 1. System Context & Constraints",
    "# 2. Functional Modules & Logic",
    "# 3. Architecture Decision Records (ADRs)",
    "# 4. Grounding Data (Few-Shot Samples)",
]

SRC_DECL_RE = re.compile(r"^\s*-\s*(SRC-\d{3})\s*:")
SRC_RE = re.compile(r"SRC-\d{3}")
FORBIDDEN_PHRASES = [
    re.compile(r"\bwork[- ]later\b", re.IGNORECASE),
]
OBJECT_ID_RE = re.compile(r"^\s*Object_ID\s*:", re.IGNORECASE)
ADR_ID_RE = re.compile(r"^\s*ADR_ID\s*:", re.IGNORECASE)
SOURCES_RE = re.compile(r"^\s*Sources\s*:", re.IGNORECASE)


def _is_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
    if not cells:
        return False
    for cell in cells:
        if not cell:
            return False
        if any(ch not in "-:" for ch in cell):
            return False
    return True


def _find_line_index(lines: list[str], target: str) -> int | None:
    for idx, line in enumerate(lines):
        if line.strip() == target:
            return idx
    return None


def validate_blueprint(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Blueprint not found: {path}"]

    text = path.read_text(encoding="utf-8")
    for phrase in FORBIDDEN_PHRASES:
        if phrase.search(text):
            errors.append("Forbidden placeholder language detected (work-later).")
            break
    if "http://" in text or "https://" in text:
        errors.append("External links are not allowed (http/https detected).")

    lines = text.splitlines()

    heading_indices: list[tuple[str, int]] = []
    for heading in REQUIRED_HEADINGS:
        idx = _find_line_index(lines, heading)
        if idx is None:
            errors.append(f"Missing required heading: {heading}")
        else:
            heading_indices.append((heading, idx))

    if len(heading_indices) == len(REQUIRED_HEADINGS):
        ordered = [idx for _heading, idx in heading_indices]
        if ordered != sorted(ordered):
            errors.append("Required headings are out of order.")

    if _find_line_index(lines, "## Validation_Checklist") is None:
        errors.append("Missing required section: ## Validation_Checklist")

    src_header = "## Source_Index:"
    cov_header = "## Coverage_Map:"
    src_idx = _find_line_index(lines, src_header)
    cov_idx = _find_line_index(lines, cov_header)

    if src_idx is None:
        errors.append(f"Missing required section: {src_header}")
    if cov_idx is None:
        errors.append(f"Missing required section: {cov_header}")

    if src_idx is not None and cov_idx is not None:
        if src_idx > cov_idx:
            errors.append("Source_Index must appear before Coverage_Map.")
        src_lines = lines[src_idx + 1 : cov_idx]
        src_ids: list[str] = []
        for line in src_lines:
            match = SRC_DECL_RE.match(line)
            if match:
                src_ids.append(match.group(1))

        if not src_ids:
            errors.append("Source_Index must list at least one SRC-### entry.")

        duplicates = sorted({src_id for src_id in src_ids if src_ids.count(src_id) > 1})
        if duplicates:
            errors.append(
                "Duplicate SRC entries in Source_Index: " + ", ".join(duplicates)
            )

        coverage_lines: list[str] = []
        for line in lines[cov_idx + 1 :]:
            if line.startswith("# "):
                break
            coverage_lines.append(line)

        coverage_ids: list[str] = []
        for line in coverage_lines:
            coverage_ids.extend(SRC_RE.findall(line))

        if not coverage_ids:
            errors.append("Coverage_Map must reference at least one SRC-### entry.")

        src_set = set(src_ids)
        unknown = sorted({src_id for src_id in coverage_ids if src_id not in src_set})
        if unknown:
            errors.append("Coverage_Map references unknown SRC IDs: " + ", ".join(unknown))

        counts = {src_id: coverage_ids.count(src_id) for src_id in src_set}
        missing = sorted([src_id for src_id, count in counts.items() if count == 0])
        if missing:
            errors.append(
                "Coverage_Map missing SRC IDs from Source_Index: " + ", ".join(missing)
            )

        multiple = sorted([src_id for src_id, count in counts.items() if count > 1])
        if multiple:
            errors.append(
                "Coverage_Map references SRC IDs more than once: " + ", ".join(multiple)
            )

    block_starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        if OBJECT_ID_RE.match(line) or ADR_ID_RE.match(line):
            block_starts.append((line.strip(), idx))

    for block_label, start_idx in block_starts:
        has_sources = False
        for line in lines[start_idx + 1 :]:
            if line.startswith("# ") or not line.strip():
                break
            if SOURCES_RE.match(line):
                has_sources = True
                break
        if not has_sources:
            errors.append(f"Missing Sources for block starting at line {start_idx + 1}: {block_label}")

    table_indices: list[int] = []
    for idx, line in enumerate(lines[:-1]):
        if "|" in line and _is_table_separator(lines[idx + 1]):
            table_indices.append(idx)

    for header_idx in table_indices:
        data_rows = 0
        for line in lines[header_idx + 2 :]:
            if not line.strip() or "|" not in line:
                break
            data_rows += 1
        if data_rows < 3:
            errors.append(
                f"Few-shot table starting at line {header_idx + 1} has fewer than 3 data rows."
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BLUEPRINT.md structure and coverage.")
    parser.add_argument(
        "--path",
        default="BLUEPRINT.md",
        help="Path to the blueprint file (default: BLUEPRINT.md).",
    )
    args = parser.parse_args()

    errors = validate_blueprint(Path(args.path))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("Blueprint validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
