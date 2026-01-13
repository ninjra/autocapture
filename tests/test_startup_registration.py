from __future__ import annotations

from autocapture.win32.startup import disable_startup, enable_startup, is_startup_enabled


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    KEY_READ = 1
    KEY_SET_VALUE = 2
    REG_SZ = 1

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    class _Key:
        def __init__(self, store: dict[str, dict[str, str]], path: str) -> None:
            self._store = store
            self._path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def CreateKey(self, root, path):  # noqa: N802 - mimic winreg API
        if path not in self._store:
            self._store[path] = {}
        return FakeWinreg._Key(self._store, path)

    def OpenKey(self, root, path, reserved=0, access=0):  # noqa: N802 - mimic winreg API
        if path not in self._store:
            raise FileNotFoundError(path)
        return FakeWinreg._Key(self._store, path)

    def SetValueEx(self, key, name, reserved, reg_type, value):  # noqa: N802
        self._store[key._path][name] = value

    def QueryValueEx(self, key, name):  # noqa: N802
        if name not in self._store[key._path]:
            raise FileNotFoundError(name)
        return self._store[key._path][name], self.REG_SZ

    def DeleteValue(self, key, name):  # noqa: N802
        if name not in self._store[key._path]:
            raise FileNotFoundError(name)
        del self._store[key._path][name]


def test_startup_registry_cycle() -> None:
    fake = FakeWinreg()
    command = "autocapture tray"

    assert not is_startup_enabled(winreg_module=fake)

    enable_startup(command, winreg_module=fake)
    assert is_startup_enabled(winreg_module=fake)

    disable_startup(winreg_module=fake)
    assert not is_startup_enabled(winreg_module=fake)
