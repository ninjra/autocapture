"""Download and extract Windows vendor binaries (Qdrant + FFmpeg)."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_QDRANT_VERSION = "1.16.3"
QDRANT_ZIP_NAME = "qdrant-x86_64-pc-windows-msvc.zip"
QDRANT_SHA_NAME = "qdrant-x86_64-pc-windows-msvc.zip.sha256"
FFMPEG_ZIP_NAME = "ffmpeg-release-essentials.zip"
QDRANT_SHA256_BY_VERSION = {
    "1.16.3": "a1159282922776a05bdeaad9e90c85d8d1ca0bc90d9e5e4add56133e15748753",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        _log(f"Using cached download: {dest}")
        return
    _log(f"Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Autocapture/1.0"})
    with urllib.request.urlopen(req) as resp:
        content_type = resp.headers.get("Content-Type", "")
        final_url = resp.geturl()
        if "text/html" in content_type:
            html = resp.read().decode("utf-8", errors="ignore")
            resolved = _extract_download_link(html)
            if resolved:
                _log(f"Resolved SourceForge redirect to: {resolved}")
                _download(resolved, dest)
                return
            raise RuntimeError(f"Failed to resolve SourceForge redirect for {url}")
        _stream_to_file(resp, dest)
        _log(f"Downloaded: {dest} ({dest.stat().st_size} bytes)")
        if final_url != url:
            _log(f"Final URL: {final_url}")


def _stream_to_file(resp, dest: Path) -> None:
    with dest.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def _extract_download_link(html: str) -> str | None:
    pattern = r"https?://[^\"']+qdrant-x86_64-pc-windows-msvc\.zip"
    match = re.search(pattern, html)
    if match:
        return match.group(0)
    pattern = r"https?://[^\"']+ffmpeg-release-essentials\.zip"
    match = re.search(pattern, html)
    if match:
        return match.group(0)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(archive: Path, sha_file: Path) -> None:
    raw = sha_file.read_text(encoding="utf-8").strip()
    expected = raw.split()[0]
    actual = _sha256(archive)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"SHA256 mismatch for {archive.name}: expected {expected}, got {actual}")
    _log(f"SHA256 OK for {archive.name}")


def _ensure_pinned_sha(downloads_dir: Path, qdrant_version: str) -> Path | None:
    expected = QDRANT_SHA256_BY_VERSION.get(qdrant_version)
    if not expected:
        return None
    downloads_dir.mkdir(parents=True, exist_ok=True)
    sha_path = downloads_dir / QDRANT_SHA_NAME
    if sha_path.exists():
        existing = sha_path.read_text(encoding="utf-8").strip()
        if existing and not existing.startswith(expected):
            raise RuntimeError(
                f"Pinned SHA256 mismatch for Qdrant {qdrant_version}: "
                f"expected {expected}, found {existing}"
            )
        return sha_path
    sha_path.write_text(f"{expected}  {QDRANT_ZIP_NAME}\n", encoding="utf-8")
    return sha_path


def _extract_qdrant(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    qdrant_exe = dest_dir / "qdrant.exe"
    if qdrant_exe.exists():
        _log(f"Qdrant already extracted: {qdrant_exe}")
        return
    _log(f"Extracting Qdrant to {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("qdrant.exe"):
                zf.extract(member, dest_dir)
                extracted = dest_dir / member
                extracted.replace(qdrant_exe)
            elif member and not member.endswith("/"):
                zf.extract(member, dest_dir)
    _log("Qdrant extraction complete")


def _extract_ffmpeg(zip_path: Path, dest_dir: Path) -> None:
    target_bin = dest_dir / "bin"
    target_bin.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = target_bin / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        _log(f"FFmpeg already extracted: {ffmpeg_exe}")
        return
    _log(f"Extracting FFmpeg to {dest_dir}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        exe_candidates = list(tmp_path.rglob("ffmpeg.exe"))
        if not exe_candidates:
            raise RuntimeError("ffmpeg.exe not found in archive")
        base_dir = exe_candidates[0].parent
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            src = base_dir / name
            if src.exists():
                shutil.copy2(src, target_bin / name)
    _log("FFmpeg extraction complete")


def _build_urls(qdrant_version: str) -> dict[str, str]:
    qdrant_base = f"https://github.com/qdrant/qdrant/releases/download/v{qdrant_version}"
    qdrant_url = f"{qdrant_base}/{QDRANT_ZIP_NAME}"
    qdrant_sha = f"{qdrant_base}/{QDRANT_SHA_NAME}"
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    ffmpeg_sha = ffmpeg_url + ".sha256"
    return {
        "qdrant": qdrant_url,
        "qdrant_sha": qdrant_sha,
        "ffmpeg": ffmpeg_url,
        "ffmpeg_sha": ffmpeg_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qdrant-version",
        default=os.environ.get("QDRANT_VERSION", DEFAULT_QDRANT_VERSION),
        help=f"Qdrant version (default {DEFAULT_QDRANT_VERSION})",
    )
    parser.add_argument(
        "--vendor-dir",
        default="vendor",
        help="Destination vendor directory (default: vendor)",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        _log("Non-Windows platform detected; vendor download skipped.")
        return 0

    vendor_dir = Path(args.vendor_dir)
    downloads_dir = vendor_dir / "_downloads"
    urls = _build_urls(args.qdrant_version)

    qdrant_zip = downloads_dir / QDRANT_ZIP_NAME
    qdrant_sha = _ensure_pinned_sha(downloads_dir, args.qdrant_version)
    if qdrant_sha is None:
        qdrant_sha = downloads_dir / QDRANT_SHA_NAME
    ffmpeg_zip = downloads_dir / FFMPEG_ZIP_NAME
    ffmpeg_sha = downloads_dir / f"{FFMPEG_ZIP_NAME}.sha256"

    _download(urls["qdrant"], qdrant_zip)
    if qdrant_sha.name == QDRANT_SHA_NAME and qdrant_sha.exists() is False:
        _download(urls["qdrant_sha"], qdrant_sha)
    _download(urls["ffmpeg"], ffmpeg_zip)
    _download(urls["ffmpeg_sha"], ffmpeg_sha)
    _verify_sha256(qdrant_zip, qdrant_sha)
    _verify_sha256(ffmpeg_zip, ffmpeg_sha)

    _extract_qdrant(qdrant_zip, vendor_dir / "qdrant")
    _extract_ffmpeg(ffmpeg_zip, vendor_dir / "ffmpeg")

    _log("Vendor binaries ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
