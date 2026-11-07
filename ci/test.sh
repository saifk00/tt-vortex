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

python -m unittest discover -s tests -v 2>&1 | tee "$out/tests.log"
python -m triton.tools.compile examples/add.py \
    --kernel-name add_kernel --signature '*i32,*i32,*i32,i32,256' \
    --grid '(n+255)/256,1,1' --num-warps 4 --target vortex:32:32 \
    --out-name add --out-path "$out/add"
mv "$out"/add.*.h "$out/kernel.h"
mv "$out"/add.*.cpp "$out/kernel.cpp"
read -r -a config_flags <<< "$CONFIGS"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -DTTVORTEX_AOT_SHORT_NAMES \
    "${config_flags[@]}" -DBENCH_BLOCK=256 -DBENCH_WARPS=4 \
    -I"$out" -I"$VORTEX_HOME/sw/runtime/include" -I"$VORTEX_BUILD_DIR/sw" \
    examples/add.cpp "$out/kernel.cpp" -L"$VORTEX_BUILD_DIR/sw/runtime" \
    -Wl,-rpath,"$VORTEX_BUILD_DIR/sw/runtime" -lvortex -o "$out/add"
for n in 1024 1001; do
    (cd "$VORTEX_BUILD_DIR"
     VORTEX_DRIVER=simx VORTEX_PROFILING=1 timeout 120 "$out/add" "$n" 3 "$out/simx-n$n") \
        2>&1 | tee "$out/simx-n$n.log"
done
