from __future__ import annotations

from pathlib import Path

from autocapture.bench import summary


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert summary.percentile(values, 0.5) == 3.0
    assert summary.percentile(values, 0.95) == 4.8


def test_parse_timings_tsv(tmp_path: Path) -> None:
    timing = tmp_path / "timing.tsv"
    timing.write_text("iteration\tms\n1\t10\n2\t20\n", encoding="utf-8")
    values = summary.parse_timings(timing)
    assert values == [10.0, 20.0]
