# Basic kernel compilation

[examples/basic.py](../examples/basic.py) exercises seven operations. All seven
compile to NVIDIA sm80 CUBIN with Triton 3.9.0. The current Vortex results below
use `vortex:32:32`, four warps, and `BLOCK=256` (a 16-by-16 tile for transpose).
Float add, block sum, prefix sum, atomic sum, and transpose have simulator regression coverage.

| Kernel | Vortex outcome |
| --- | --- |
| `multiply` | Produces LLVM IR, VXBIN, and the AOT launcher. |
| `gather` | Produces LLVM IR, VXBIN, and the AOT launcher. |
| `float_add` | Compiles and simulates correctly for f32, including IEEE edge cases. |
| `block_sum` | Produces LLVM IR, VXBIN, and the AOT launcher; masked sums verified on SimX. |
| `prefix_sum` | Compiles and simulates correctly for i32; inclusive scan restarts each tile. |
| `atomic_sum` | Compiles and simulates correctly for relaxed i32 atomic addition. |
| `transpose` | Compiles and simulates correctly for out-of-place i32/f32 matrix transpose. |

Float addition registers Triton's
`ElementwiseOpConversion<arith::AddFOp, LLVM::FAddOp>`, as NVIDIA does. This
unpacks each thread's LLVM struct, emits scalar `llvm.fadd` operations, and
repacks the result. The existing type converter and masked loads/stores already
handle f32. The SDK's RV32 F extension supplies native float arithmetic.
Float reductions, scans, and atomics remain unsupported; this does not enable
arbitrary floating-point operations or fp16/fp64 tensors.

Transpose uses rank-two blocked tensors and sliced layouts, now accepted by
[TritonGPUToLLVM.cpp](../lib/TritonVortexGPUToLLVM/TritonGPUToLLVM.cpp).
Triton's generic view patterns lower expand-dims, broadcast, and `tt.trans`;
`tt.trans` changes the coordinate/layout interpretation without moving registers.
The following `ttg.convert_layout` redistributes values to the output owners.
Triton's linear-layout machinery computes the shared-memory addresses, and
our existing target hooks emit the loads, stores, and CTA barriers. No special
case replaces the kernel with a different transpose algorithm.
Within-warp conversions can instead use indexed shuffles. For f32 payloads the
target hook bitcasts to i32, shuffles integer registers, and bitcasts back;
there is no numerical conversion.

Shared memory here is an explicitly addressed per-CTA scratchpad, separate from
the L1 data cache. Its base comes from CSR `0xcdf`; the compiler reports its size
in `ttg.shared` for the launcher. The 16-by-16, four-warp fixture needs 2048 bytes,
including scratch for pointer and mask layout conversions. A barrier makes
shared writes visible before other threads read them; Triton's allocation and
memory-barrier analysis also protects scratch reuse. This is an out-of-place
transpose with separate input and output buffers, not an in-place algorithm.

The reduction lowering uses Triton's generic reduce and layout-conversion
patterns with Vortex target hooks. For `BLOCK=256`, four 32-lane warps first
combine two registers per thread, then reduce within each warp using XOR
shuffles. A shared-memory exchange uses 16 bytes of CTA-local scratch; two
more XOR shuffles combine the four warp totals. One thread stores the scalar
result. Shuffle assembly is marked as having side effects, matching Vortex's
volatile intrinsics, so LLVM keeps it outside the single-thread output branch.
Reduction elements currently must be `i32`.

Scan uses Triton's generic scan patterns and the existing Vortex shuffle/shared
memory hooks. With four warps and `BLOCK=256`, it uses 32 bytes of shared scratch.
Atomic add has a scalar LLVM `atomicrmw` rewrite, with masks and redundant-owner
predicates guarding each update. Consumed old values are broadcast through
Triton's atomic-result utilities. The SDK emits `amoadd.w`; atomics currently
support only relaxed i32 addition, using LLVM system scope.

The simulator must be built with `-DVX_CFG_EXT_A_ENABLE`. The pinned CI SDK now
includes this flag; Vortex's default configuration disables atomics. After
changing the SDK manifest, rebuild it with `ci/prepare_vortex.py` in a fresh
prefix (or remove the old generated SDK first), then source its `env.sh`.

`block_sum` writes one sum per program; `prefix_sum` starts over each program;
`atomic_sum` accumulates into one output word (initialize it to zero for a sum).
To check generated IR and actual simulated results:

```bash
make check-ir
source .cache/vortex-sdk/env.sh
VORTEX_DRIVER=simx python -m unittest discover -s tests -p test_basic_runtime.py -v
```

The simulator tests launch the actual `examples/basic.py` kernels, comparing
against NumPy for empty and partial tiles, one/four warps, multiple CTAs, signed
wraparound, untouched output guards, and repeated accumulation. Additional
kernels check returned atomic tickets and scalar-result broadcasts. Float-add
checks exact non-NaN result bits (including rounding, signed zero, subnormals,
and overflow) and NaN classification against NumPy.

Transpose checks 600 launches: i32/f32 bit patterns, ten matrix shapes (including
empty dimensions), five square/rectangular tiles, one/two/four warps, and two
repeats. Extra fully masked CTAs, untouched output guards, and unchanged input
buffers catch indexing errors and unwanted writes. Float values are compared
as raw bits, preserving NaN payloads and signed zero.

From the tt-vortex repository with the venv activated:

```bash
source .cache/vortex-sdk/env.sh
bash examples/compile_basic.sh
# Or inspect one case:
bash examples/compile_basic.sh block_sum
```

The shell helper calls the standard `python -m triton.tools.compile` CLI for
each kernel and continues after compilation failures. Inspect
`build/basic/<kernel>/compile.log` for the full diagnostic and pass dump,
and `<kernel>.ttir` / `<kernel>.ttgir` beside it for the completed stages.
Successful cases also have `<kernel>.llir`, `<kernel>.vxbin`, and generated
launcher/header files. `build/basic/results.csv` records the exit codes from
the most recent invocation.

To inspect the lowered LLVM-dialect MLIR with the native optimizer:

```bash
build/bin/ttvortex-opt tests/ir/block_sum.ttgir \
  --convert-triton-gpu-to-vortex-llvm \
  --reconcile-unrealized-casts --canonicalize --cse \
  -o build/block_sum.lowered.mlir
```
