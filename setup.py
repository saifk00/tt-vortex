"""Build the native Triton plugin for wheels and editable installs."""

import os
from pathlib import Path
import platform
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise RuntimeError("tt-vortex currently supports Linux x86-64")
        # Build-time LLVM headers/TableGen are separate from the runtime Vortex SDK.
        sdk = Path(os.environ.get("LLVM_SYSPATH", ROOT / ".cache/llvm")).resolve()
        subprocess.run([sys.executable, str(ROOT / "ci/fetch_llvm.py"), str(sdk)], check=True)
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build = Path(self.build_temp).resolve() / ext.name
        subprocess.run([
            "cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={sys.executable}",
            f"-DLLVM_SYSPATH={sdk}", f"-DTTVORTEX_OUTPUT_DIRECTORY={output}",
            # Installed wheels find exactly the pinned Triton package beside them.
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        ], check=True)
        subprocess.run(["cmake", "--build", str(build), "--parallel",
                        os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2")], check=True)


setup(
    ext_modules=[Extension("tt_vortex._C", sources=[], py_limited_api=True)],
    cmdclass={"build_ext": CMakeBuild},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
