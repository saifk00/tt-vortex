"""Run native MLIR fixtures with the pinned LLVM testing tools."""

import os
from pathlib import Path

import lit.formats
import lit.llvm
from lit.llvm.subst import ToolSubst

root = Path(__file__).resolve().parents[2]
llvm = Path(os.environ.get("LLVM_SYSPATH", root / ".cache/llvm")).resolve()
opt = Path(os.environ.get("TTVORTEX_OPT", root / "build/bin/ttvortex-opt")).resolve()
if not opt.is_file():
    lit_config.fatal(f"Missing {opt}; build it with make opt")

config.name = "tt-vortex"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".ttir", ".ttgir", ".mlir"]
config.test_source_root = str(Path(__file__).parent)
config.test_exec_root = str(root / "build/lit")
config.excludes = ["Inputs"]
config.environment["FILECHECK_OPTS"] = "--enable-var-scope"
# Only the native optimizer runs; backend-discovery settings must not leak in.
for key in list(config.environment):
    if key.startswith(("TRITON_", "VORTEX_")):
        del config.environment[key]

lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config

llvm_config.add_tool_substitutions(
    [ToolSubst("ttvortex-opt", command=str(opt), unresolved="fatal"),
     ToolSubst("FileCheck", unresolved="fatal"),
     ToolSubst("not", unresolved="fatal"),
     ToolSubst("split-file", unresolved="fatal")],
    [str(llvm / "bin")],
)
