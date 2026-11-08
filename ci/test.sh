#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VORTEX_HOME:?Set VORTEX_HOME or source .cache/vortex-sdk/env.sh}"
: "${VORTEX_BUILD_DIR:?Set VORTEX_BUILD_DIR}"
: "${CONFIGS:?Use the same CONFIGS as the simulator build}"
export TRITON_DEFAULT_BACKEND=vortex TRITON_STORE_BINARY_ONLY=0
out="$PWD/build/ci"
mkdir -p "$out"
export TRITON_CACHE_DIR="$out/cache"
export TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$out/ir"

timeout 600 python -m unittest discover -s tests -v 2>&1 | tee "$out/tests.log"
VORTEX_DRIVER=simx AOT_OUT_DIR="$out/aot" bash examples/run_aot.sh 1024 1001
for n in 1024 1001; do
    VORTEX_DRIVER=simx timeout 120 python examples/add.py --n "$n" --repeats 3 \
        --output-prefix "$out/jit-simx-n$n" 2>&1 | tee "$out/jit-simx-n$n.log"
    python ci/compare_add.py "$out/jit-simx-n$n" "$out/aot/simx-n$n" \
        2>&1 | tee "$out/compare-simx-n$n.log"
done
