from pathlib import Path

from autocapture.runtime_env import ProfileName
from autocapture.runtime_profile_override import read_profile_override, write_profile_override


def test_profile_override_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "profile_override.json"
    exists, profile = read_profile_override(path)
    assert exists is False
    assert profile is None

    write_profile_override(path, ProfileName.FOREGROUND, source="test")
    exists, profile = read_profile_override(path)
    assert exists is True
    assert profile == ProfileName.FOREGROUND

    write_profile_override(path, None, source="test")
    exists, profile = read_profile_override(path)
    assert exists is True
    assert profile is None
