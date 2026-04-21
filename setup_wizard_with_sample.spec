# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for setup_wizard_with_sample.py (console installer + optional sample batch).
# Build: pip install pyinstaller requests && pyinstaller setup_wizard_with_sample.spec

a = Analysis(
    ["setup_wizard_with_sample.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "requests",
        "urllib3",
        "certifi",
        "charset_normalizer",
        "idna",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="setup_wizard_with_sample",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
