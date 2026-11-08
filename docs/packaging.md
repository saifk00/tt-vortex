# Packaging

`tt-vortex` ships the Python backend/runtime, native lowering extension, and
Triton AOT templates. It depends on exactly `triton==3.9.0`: CPython's stable ABI
does not make the LLVM/MLIR interface stable across Triton releases.

```bash
python -m pip install build
python -m build
```

This builds a source archive, then a wheel from that archive. The isolated build
installs its Python build dependencies and downloads the pinned LLVM headers and
TableGen SDK. Set `LLVM_SYSPATH` to reuse an extracted SDK. Only our extension is
compiled; Triton and LLVM are not rebuilt. `make wheel` reuses `.cache/llvm`.

Wheels use `cp312-abi3` and Linux x86-64 platform tags. CI runs `auditwheel` with
a glibc 2.39 baseline, then tests on Python 3.12 and 3.13. The extension locates
`libtriton.so` relative to the installed package; that library comes from the
pinned Triton dependency and must not be bundled a second time. Supporting older
Linux distributions will require building and testing against an older baseline.

CI installs the wheel in fresh virtual environments and copies only the tests
and examples to a temporary directory. It checks backend discovery, native
loading, packaged AOT templates, and JIT/AOT simulation with exact counter/output
comparisons. To repeat this locally with an existing SDK:

```bash
source .cache/vortex-sdk/env.sh
python ci/test_wheel.py dist/tt_vortex-*.whl
```

## External SDK

The package does not yet distribute or download the Vortex SDK at runtime.
`VORTEX_SDK_PATH=/path/to/sdk` selects a directory with `source/`, `tools/`, and
`build/`, matching `ci/prepare_vortex.py`. Existing component overrides take
precedence. No checkout-relative paths are assumed.

A separate, pinned SDK wheel is a likely distribution model: a future
`tt-vortex[simx]` extra could install it, with discovery handled in
`backend/sdk.py`. This allows the large toolchain/runtime bundle to evolve
separately from the frontend. Numba-CUDA uses this pattern:
[`numba-cuda[cu12]`](https://nvidia.github.io/numba-cuda/user/installation.html)
installs its CUDA toolkit dependencies from PyPI and also supports external
toolkits. We have not added that extra or chosen an SDK packaging format yet.

The current CI produces downloadable distributions; it does not publish to PyPI.
