import json
import sys

from autocapture import main as main_module


def test_print_config_json(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "autocapture.yml"
    config_path.write_text("offline: false\n", encoding="utf-8")

    monkeypatch.setenv("AUTOCAPTURE_TEST_MODE", "1")
    monkeypatch.setenv("AUTOCAPTURE_GPU_MODE", "off")

    monkeypatch.setattr(
        sys, "argv", ["autocapture", "--config", str(config_path), "print-config", "--json"]
    )
    main_module.main()
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert isinstance(payload, dict)
    assert "database" in payload
