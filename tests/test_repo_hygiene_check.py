from __future__ import annotations

from tools.repo_hygiene_check import find_violations


def test_repo_hygiene_flags_build_and_ide_artifacts() -> None:
    paths = [
        "src/Autocapture.Shell/obj/Debug/net8.0-windows/file.cs",
        "src/Autocapture.Shell/bin/Release/net8.0-windows/app.exe",
        ".idea/workspace.xml",
        "src/Autocapture.Shell/Program.cs",
    ]

    violations = find_violations(paths)

    assert "src/Autocapture.Shell/obj/Debug/net8.0-windows/file.cs" in violations
    assert "src/Autocapture.Shell/bin/Release/net8.0-windows/app.exe" in violations
    assert ".idea/workspace.xml" in violations
    assert "src/Autocapture.Shell/Program.cs" not in violations
