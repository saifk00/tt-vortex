# Debugging the MLIR lowering

`ttvortex-opt` is a native MLIR optimizer driver for TTIR and TTGIR. It shares
the lowering object code with the Python extension and links against the pinned
Triton wheel. It uses MLIR's standard parser, verifier, pass-pipeline interface,
IR dumps, and bytecode support. No Vortex toolchain or simulator is needed.

From the tt-vortex repository with the development venv activated:

```bash
make opt BUILD_TYPE=Debug
build/bin/ttvortex-opt --help
```

This builds `build/bin/ttvortex-opt`. The additional native build dependency is
the matching Python development library: `libtriton.so` contains Python bindings
whose symbols must be resolved, although this executable does not start a Python
interpreter. The existing pinned LLVM SDK supplies the optimizer-driver libraries.
The binary uses the installed Triton wheel's library path; rebuild if that
environment moves. This is currently a development build target, not a wheel
entry point.

## Step through a pass

The checked-in fixtures can be used directly, without running Python first:

```bash
lldb \
  -o 'breakpoint set --func-regex "ConvertTritonGPUToVortexLLVM::runOnOperation"' \
  -o run \
  -- build/bin/ttvortex-opt tests/ir/float_add.ttgir \
  --convert-triton-gpu-to-vortex-llvm \
  --mlir-disable-threading --mlir-print-ir-after-failure
```

Use `next`, `step`, `frame variable`, and `bt` in LLDB.
`expression -- op->dump()` prints an operation while stopped in the pass.
Use `tests/ir/transpose.ttgir` to follow rank-two layout conversion, and add
`--mlir-print-ir-before-all --mlir-print-ir-after-all` to inspect pass boundaries.
Clang Debug builds use `-fstandalone-debug` to retain the MLIR type information
for debugger expressions.

Our lowering is compiled with debug information. Functions supplied by the
prebuilt Triton wheel remain optimized; this does not rebuild Triton with debug
symbols. No debugpy session is needed.

## Run the stages separately

```bash
build/bin/ttvortex-opt build/basic/multiply/multiply.ttir \
  '--convert-triton-to-tritongpu=target=vortex:32 num-warps=4 threads-per-warp=32 num-ctas=1' \
  --canonicalize --cse -o build/basic/multiply/opt.ttgir

build/bin/ttvortex-opt build/basic/multiply/opt.ttgir \
  '--pass-pipeline=builtin.module(convert-scf-to-cf,convert-triton-gpu-to-vortex-llvm,reconcile-unrealized-casts,canonicalize,cse,symbol-dce)' \
  -o build/basic/multiply/opt.llvm.mlir
```

These reproduce the pass sequences in the Python backend's `make_ttgir` and
`make_llir` up to MLIR conversion. The output is **LLVM-dialect MLIR**, not LLVM
IR text. LLVM IR translation, target metadata, launch-wrapper insertion, and
Vortex SDK code generation are outside this tool. The input filename does not
select a pipeline: without pass flags, the driver parses, verifies, and prints.

```bash
python -m unittest discover -s tests -p test_opt.py -v
```

The native tests cover TTIR → TTGIR → LLVM-dialect MLIR, shared-memory reduction lowering,
and bytecode round-tripping, without invoking the Python backend.

For lowering regressions, add TTIR/TTGIR snippets with `RUN:` and `CHECK:`
comments to [tests/ir](../tests/ir/README.md). `make check-ir` builds the optimizer
and runs those files with lit/FileCheck; CI runs the same target.
