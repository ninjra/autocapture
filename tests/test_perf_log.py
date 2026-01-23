from pathlib import Path

from autocapture.ux.perf_log import read_perf_log


def test_read_perf_log_tail(tmp_path: Path) -> None:
    data_dir = tmp_path
    perf_dir = data_dir / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    path = perf_dir / "runtime.jsonl"
    lines = [f'{{"time_utc":"t{i}","component":"runtime"}}' for i in range(5)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = read_perf_log(data_dir, component="runtime", limit=3)

    entries = payload["entries"]
    assert len(entries) == 3
    assert entries[0]["parsed"]["time_utc"] == "t2"
    assert entries[-1]["parsed"]["time_utc"] == "t4"
