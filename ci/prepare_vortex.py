"""Build the pinned SimX SDK once; subsequent CI jobs restore this directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def checkout(repository, commit, destination):
    if not (destination / ".git").exists():
        run("git", "init", destination)
        run("git", "-C", destination, "fetch", "--depth=1", repository, commit)
        run("git", "-C", destination, "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Unexpected revision in {destination}: {actual}, expected {commit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path(".cache/vortex-sdk"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    root = args.prefix.resolve()
    manifest = Path(__file__).with_name("dependencies.lock.json")
    spec = json.loads(manifest.read_text())["vortex"]
    source, build, tools = root / "source", root / "build", root / "tools"
    settings = {
        "VORTEX_HOME": str(source),
        "VORTEX_BUILD_DIR": str(build),
        "VORTEX_TOOLDIR": str(tools),
        "CONFIGS": " ".join(f"-D{key}={value}" for key, value in spec["simulator"]["defines"].items()),
    }
    # Generated Makefiles and simulator rpaths contain absolute paths.
    signature = hashlib.sha256(manifest.read_bytes() + Path(__file__).read_bytes() + str(root).encode()).hexdigest()
    stamp = root / "ready"
    if stamp.exists() and stamp.read_text().strip() != signature:
        raise RuntimeError(f"SDK inputs or location changed; remove {root} and rerun")
    if not stamp.exists():
        checkout(spec["repository"], spec["commit"], source)
        run("git", "-C", source, "submodule", "update", "--init", "--depth=1",
            "third_party/softfloat", "third_party/ramulator")
        ramulator = source / "third_party/ramulator"
        cmake_sources = []
        for name, dependency in spec["ramulator_dependencies"].items():
            path = ramulator / "ext" / name
            checkout(dependency["repository"], dependency["commit"], path)
            cmake_sources.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={path}")
        build.mkdir(parents=True, exist_ok=True)
        run(source / "configure", f"--xlen={spec['simulator']['xlen']}",
            f"--osversion={spec['toolchain']['osversion']}", f"--tooldir={tools}", cwd=build)
        tool_env = dict(os.environ, TOOLCHAIN_REV=spec["toolchain"]["commit"])
        for component, option in (("llvm-vortex", "--llvm"), ("riscv32-gnu-toolchain", "--riscv32")):
            if not (tools / component).is_dir():
                run(build / "ci/toolchain_install.sh", option, cwd=build, env=tool_env)
        run("cmake", "-S", ramulator, "-B", ramulator / "build", *cmake_sources)
        run("cmake", "--build", ramulator / "build", "--target", "ramulator", "--parallel", args.jobs)
        run("make", "-C", source / "third_party", f"-j{args.jobs}", "softfloat", "ramulator")
        # Vortex uses -Werror; external fmt headers trigger newer compiler diagnostics.
        # Classify third-party includes correctly while retaining errors in Vortex.
        external_headers = [source / "third_party/softfloat/source/include",
                            ramulator / "ext/spdlog/include", ramulator / "ext/yaml-cpp/include",
                            ramulator / "src"]
        # Exporting CXXFLAGS makes runtime-only resolved macros leak into SimX's
        # recursive make. Put common system includes in the compiler command instead.
        compiler = "g++ " + " ".join(f"-isystem {shlex.quote(str(path))}" for path in external_headers)
        runtime_env = dict(os.environ)
        runtime_env.pop("CXXFLAGS", None)
        run("make", "-C", build / "sw/runtime", f"-j{args.jobs}", f"CXX={compiler}", "DEBUG=",
            f"PERF={1 if spec['simulator']['performance_counters'] else ''}",
            f"CONFIGS={settings['CONFIGS']}", "stub", spec["simulator"]["driver"], env=runtime_env)
        stamp.write_text(signature + "\n")
    else:
        print(f"Reusing pinned SimX SDK: {root}")
    (root / "env.sh").write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()))
    if args.github_env:
        with args.github_env.open("a") as output:
            for key, value in settings.items():
                output.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
