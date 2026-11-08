"""Locate an external SDK without depending on the backend's checkout location."""

import os
from pathlib import Path


def sdk_paths():
    """Return (source, tools, LLVM, build); explicit component overrides win.

    VORTEX_SDK_PATH names a directory containing source/, tools/ and build/,
    matching ci/prepare_vortex.py. VORTEX_HOME supports existing source builds.
    No downloading or building occurs while importing or using the package.
    """
    root = os.environ.get("VORTEX_SDK_PATH")
    home = os.environ.get("VORTEX_HOME")
    if not root and not home:
        raise RuntimeError(
            "Vortex SDK not configured. Set VORTEX_SDK_PATH to an SDK containing "
            "source/, tools/ and build/, or set VORTEX_HOME, VORTEX_TOOLDIR and "
            "VORTEX_BUILD_DIR for an existing source build. The SDK is installed separately."
        )
    root = Path(root).expanduser().resolve() if root else None
    home = Path(home).expanduser().resolve() if home else root / "source"
    tools = Path(os.environ.get("VORTEX_TOOLDIR", root / "tools" if root else home / "tools")).expanduser().resolve()
    llvm = Path(os.environ.get("VORTEX_LLVM_PATH", tools / "llvm-vortex")).expanduser().resolve()
    build = Path(os.environ.get("VORTEX_BUILD_DIR", root / "build" if root else home / "build32")).expanduser().resolve()
    return home, tools, llvm, build
