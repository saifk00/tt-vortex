"""Test an installed wheel in a fresh venv, with only test/example sources copied in."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=Path("build/ci"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel, output = args.wheel.resolve(), args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="tt-vortex-installed-") as directory:
        work = Path(directory)
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHON", "TRITON_PLUGIN_DIRS"):
            env.pop(name, None)
        subprocess.run([args.python, "-m", "venv", str(work / "venv")], check=True, env=env)
        python = work / "venv/bin/python"
        env["PATH"] = str(python.parent) + os.pathsep + env["PATH"]
        subprocess.run([str(python), "-m", "pip", "install", "--only-binary=:all:", str(wheel)],
                       cwd=work, env=env, check=True)
        # Do not copy backend/, tools/, native sources, or editable-install metadata.
        for name in ("tests", "examples"):
            shutil.copytree(root / name, work / name, ignore=shutil.ignore_patterns("__pycache__"))
        (work / "ci").mkdir()
        for name in ("test.sh", "compare_add.py"):
            shutil.copy2(root / "ci" / name, work / "ci" / name)
        subprocess.run([str(python), "-c", '''
from pathlib import Path
from importlib.resources import files
from importlib.metadata import distribution
import sys
import tt_vortex
from tt_vortex.compiler import native
from triton._C import libtriton
for module in (tt_vortex, native(), libtriton):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix)), path
    print(f"Loaded {module.__name__}: {path}")
import subprocess
dynamic = subprocess.check_output(['readelf', '-d', native().__file__], text=True)
paths = [line for line in dynamic.splitlines() if '(RUNPATH)' in line or '(RPATH)' in line]
assert len(paths) == 1 and paths[0].split('[')[1].split(']')[0] == '$ORIGIN/../triton/_C', paths
assert 'Root-Is-Purelib: false' in distribution('tt-vortex').read_text('WHEEL')
for suffix in ('cpp', 'h'):
    assert files('triton.tools.extra.vortex').joinpath('compile.' + suffix).is_file()
'''], cwd=work, env=env, check=True)
        try:
            subprocess.run(["bash", "ci/test.sh"], cwd=work, env=env, check=True)
        finally:
            results = work / "build/ci"
            if results.exists():
                shutil.copytree(results, output, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
