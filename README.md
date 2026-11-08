# tt-vortex

[![CI](https://github.com/saifk00/tt-vortex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saifk00/tt-vortex/actions/workflows/ci.yml)

A Triton backend for the [Vortex RISC-V GPU](https://github.com/vortexgpgpu/vortex).
The aim is to make the whole path from a Python kernel to simulated GPU execution
small enough to follow. Integer copy and add work through both Python JIT launches
and generated AOT launchers, on SimX and processor RTL simulation.

This is still an experimental compiler. It supports one-dimensional blocked
tensors, f32 addition, i32 reductions and scans, and relaxed i32 atomic addition.
Reductions and scans use Vortex shuffles and shared memory. Transpose remains
unsupported, and codegen is deliberately simple.

Binary wheels include the native lowering and AOT templates; the Vortex SDK is
still a separate dependency. Set `VORTEX_SDK_PATH` to its root (`source/`, `tools/`,
`build/`), or use the existing `VORTEX_HOME`, `VORTEX_TOOLDIR`, and
`VORTEX_BUILD_DIR` settings. See [packaging](docs/packaging.md) for building wheels.

## Try it

Use Linux x86-64 with Python 3.13, uv, Make, GCC and Clang installed.
From this repository:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r ci/requirements.txt triton==3.9.0
make install

python ci/prepare_vortex.py --jobs 4
source .cache/vortex-sdk/env.sh

python examples/add.py --n 1001 --output-prefix build/jit-add
bash examples/run_aot.sh 1001
```

Setup downloads the compiler toolchains and builds a SimX runtime with one core
and four 32-thread warps. Dependency revisions are pinned in
[ci/dependencies.lock.json](ci/dependencies.lock.json). The first build takes a
few minutes; subsequent native edits only need `make`. Python edits are picked
up by the editable install, and launcher-template edits need `make install-templates`.

[add.py](examples/add.py) contains the kernel and Python host code.
[run_aot.sh](examples/run_aot.sh) compiles that same kernel with Triton's standard
AOT CLI, then builds [add.cpp](examples/add.cpp) against the generated launcher.
The resulting executable embeds the VXBIN and runs without Python.

`--block` is elements per Triton program, not elements per thread. With 32-thread
warps, `BLOCK=256` and `num_warps=4` gives each thread two elements. Use at most
the hardware's four warps per core. Small tiles can have several physical
threads representing the same element; stores and atomics suppress duplicate owners.

## How it works

```text
Triton Python → TTIR → TTGIR → RISC-V LLVM → Vortex Clang → VXBIN
```

The backend is a separate package. Its native extension links against the
published Triton wheel, so there is no Triton fork or full compiler rebuild.
[compiler.py](backend/compiler.py) defines the stages; the
[native lowering](lib/TritonVortexGPUToLLVM/TritonGPUToLLVM.cpp) reuses Triton's
lowering helpers and adds Vortex-specific operations. For example, program ID X
becomes a read of CSR `0xcd6` instead of an NVIDIA special-register intrinsic.

For pass development, `make opt` builds a native `ttvortex-opt` that reads TTIR
and TTGIR directly. See [debugging](docs/debugging.md) for LLDB and pipeline commands.

Both launch paths use Vortex's device memory and command queue. The
[Python runtime](backend/runtime.py) provides typed buffers and NumPy transfers;
the AOT launcher packs the same kernel arguments through the C API. Transfers
finish before returning; kernel launches are queued, with `synchronize()` waiting
for completion. There is no Torch device integration yet.

VXBINs currently share a fixed load address. Switching kernels waits for pending
work and replaces the resident image. Use one device context at a time.

## Measuring and checking

The add examples check every output and guard element, then report cycles,
instructions, stalls, divergence, and cache activity. The Python runtime exposes
these through `device.snapshot()` and `after.delta(before)`. `--output-prefix`
saves CSVs and the final output buffer. Detailed counters need a `PERF=1` runtime,
as the setup above builds.

```bash
make check-ir
bash ci/test.sh
```

FileCheck tests cover the native lowering; transpose is marked as an expected
failure until its layout support lands. The Python tests also simulate the
[basic kernels](examples/basic.py), including partial tiles and returned atomic values.
The add checks exercise both launch paths, checking that their outputs and all
38 counter deltas agree. Both examples use the same buffers, uploads, warm-up,
and measurement boundaries; host wall-clock timings are excluded. CI runs this
on SimX from an installed wheel in fresh Python 3.12 and 3.13 environments, and
saves the distributions, logs, IR, launchers, and measurements as artifacts.

RTL is also supported with a matching runtime build: use `--driver rtlsim` for
the Python example or `VORTEX_DRIVER=rtlsim` for the AOT script.
