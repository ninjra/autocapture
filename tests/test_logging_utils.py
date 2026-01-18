import sys

import pytest

from autocapture import logging_utils


def test_default_log_dir_respects_xdg_state_home(monkeypatch, tmp_path):
    if sys.platform == "win32":
        pytest.skip("XDG state paths apply to non-Windows platforms")

    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    assert logging_utils._default_log_dir() == state_home / "autocapture" / "logs"


def test_configure_logging_skips_file_logging_on_error(monkeypatch, tmp_path):
    def _raise(*_args, **_kwargs):
        raise PermissionError("blocked")

    monkeypatch.setattr(logging_utils.Path, "mkdir", _raise)

    # Should not raise even if the log directory cannot be created.
    logging_utils.configure_logging(log_dir=tmp_path / "logs")


def test_logging_redacts_secrets(capsys):
    logging_utils.configure_logging(log_dir=None, level="INFO")
    log = logging_utils.get_logger("test")
    log.info("Bearer sk-test-secret")
    log.info("api_key=sk-test-secret")
    captured = capsys.readouterr()
    assert "sk-test-secret" not in captured.out
    assert "[REDACTED]" in captured.out
