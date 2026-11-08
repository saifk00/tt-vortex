# IR regression tests

Each `.ttir`, `.ttgir`, or `.mlir` file is a lit test. `RUN:` comments invoke
`ttvortex-opt`; FileCheck matches the resulting IR against `CHECK:` comments in
the same file. Adding a file here automatically adds it to CI.

From the repository with the development venv activated:

```bash
make check-ir
make check-ir IR_TESTS=tests/ir/program_id.ttgir
# Once the optimizer is built:
lit -v tests/ir/multiply.ttgir
```

`lit` is pinned in `ci/requirements.txt`. FileCheck, `not`, and `split-file`
come from the pinned LLVM SDK. The tests need neither the Vortex SDK nor a
simulator. Set `LLVM_SYSPATH` or `TTVORTEX_OPT` to select other build locations
when invoking lit directly. Temporary outputs go in `build/lit/`.

For example, a TTGIR test can check the CSR read produced by `tt.get_program_id`:

```mlir
// RUN: ttvortex-opt %s --convert-triton-gpu-to-vortex-llvm --canonicalize --cse | FileCheck %s
// CHECK-LABEL: llvm.func @program_x
// CHECK: %[[CSR:.*]] = llvm.mlir.constant(3286 : i32)
// CHECK: %[[ID:.*]] = llvm.inline_asm asm_dialect = att "csrr $0, $1", "=r,i" %[[CSR]] : (i32) -> i32
// CHECK: llvm.return %[[ID]] : i32
```

See `program_id.ttgir` for the complete input. Use captures like `[[ID]]` to
check data flow without fixing SSA names. `CHECK-LABEL` separates functions;
`CHECK-NEXT` checks adjacent lines when adjacency matters. Keep checks focused
on the behavior being tested instead of copying the entire generated module.

## Coverage

All fixtures must compile and satisfy their FileCheck assertions; no expected
failures remain.

| Test | Expected lowering |
| --- | --- |
| `float_add` | Two scalar f32 additions per thread, masked loads and stores, sum-to-store data flow. |
| `prefix_sum` | Upward shuffles and addition, shared partial sums and a CTA barrier, both output elements. |
| `atomic_sum` | Two masked monotonic atomic adds to the same output word, using the loaded values. |
| `transpose` | Source/destination strides, x/y program IDs, shared exchange and barrier for the transposed layout. |
| `transpose_exchange` | Isolated payload exchange: f32 bitcasts around indexed shuffles, and i32 shared stores/barrier/loads connected to the returned registers. |

Scan and transpose checks target Triton's generic lowering algorithms using our
existing Vortex shuffle/shared-memory hooks. These are structural codegen
contracts; `tests/test_basic_runtime.py` also verifies float-add, scan, atomic-add,
and transpose results on SimX, including IEEE float edge cases, partial tiles,
overflow, returned atomic values, and shared-memory reuse. Transpose is checked
bit-for-bit for i32/f32, rectangular tiles, and one/two/four warps.
Multiply, gather, and block sum already compile; block sum also has a passing IR
test here. The native driver tests also cover parsing and bytecode round-tripping.

The four inputs are unedited optimizer output from `examples/basic.py`, using
four warps and `BLOCK=256` (16 by 16 for transpose). To regenerate them:

```bash
# Obtain frontend TTIR using the ordinary AOT compiler. It retains TTIR even
# when a later stage fails. This step uses the configured development SDK.
source .cache/vortex-sdk/env.sh
BASIC_OUT_DIR=build/basic bash examples/compile_basic.sh

for kernel in float_add prefix_sum atomic_sum transpose; do
  build/bin/ttvortex-opt build/basic/$kernel/$kernel.ttir \
    '--convert-triton-to-tritongpu=target=vortex:32 num-warps=4 threads-per-warp=32 num-ctas=1' \
    --canonicalize --cse -o build/basic/$kernel/opt.ttgir
done
```

Copy the resulting module under its test's comments when intentionally updating
an input. Do not regenerate the expected checks from failing output. Running
these checked-in fixtures only needs `ttvortex-opt` and FileCheck:

```bash
make check-ir IR_TESTS=tests/ir/float_add.ttgir
# Debug the same input directly:
build/bin/ttvortex-opt tests/ir/float_add.ttgir \
  --convert-triton-gpu-to-vortex-llvm \
  --mlir-disable-threading --mlir-print-ir-after-failure
```

CI runs `make check-ir` and saves the xUnit report and lit output files. Both a
compiler failure and missing expected output fail the job.
