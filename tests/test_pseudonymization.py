from pathlib import Path

from autocapture.memory.entities import SecretStore, stable_token


def test_stable_token_is_deterministic(tmp_path: Path) -> None:
    store = SecretStore(tmp_path)
    key = store.get_or_create()
    token_a = stable_token("ORG_", "example", key)
    token_b = stable_token("ORG_", "example", key)
    assert token_a == token_b
    assert token_a.startswith("ORG_")
