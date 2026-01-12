# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

def vendor_datas(root, prefix):
    datas = []
    for dirpath, _, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        dest_dir = prefix if rel_dir == "." else os.path.join(prefix, rel_dir)
        for filename in filenames:
            datas.append((os.path.join(dirpath, filename), dest_dir))
    return datas

a = Analysis(
    ["autocapture/main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("alembic.ini", "."),
        ("alembic", "alembic"),
        ("autocapture.yml", "."),
        ("autocapture/prompts/derived/*.yaml", "autocapture/prompts/derived"),
        ("autocapture/ui/web", "autocapture/ui/web"),
    ]
    + vendor_datas("vendor/ffmpeg", "ffmpeg")
    + vendor_datas("vendor/qdrant", "qdrant"),
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="autocapture",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="autocapture",
)
