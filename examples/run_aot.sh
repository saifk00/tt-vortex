#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VORTEX_HOME:?Set VORTEX_HOME or source .cache/vortex-sdk/env.sh}"
: "${VORTEX_BUILD_DIR:?Set VORTEX_BUILD_DIR}"
: "${CONFIGS:?Use the same CONFIGS as the simulator build}"
block=${BLOCK:-256}
warps=${WARPS:-4}
driver=${VORTEX_DRIVER:-simx}
repeats=${REPEATS:-3}
if [[ ! $block =~ ^[1-9][0-9]*$ || ! $warps =~ ^[1-9][0-9]*$ ]]; then
    echo 'BLOCK and WARPS must be positive decimal integers' >&2
    exit 1
fi
mkdir -p "${AOT_OUT_DIR:-build/aot-b${block}-w${warps}}"
out=$(cd "${AOT_OUT_DIR:-build/aot-b${block}-w${warps}}" && pwd)
export TRITON_DEFAULT_BACKEND=vortex TRITON_STORE_BINARY_ONLY=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$out/cache}"
export TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$out/ir"

# The CLI suffix identifies the specialization; the host includes a stable filename.
rm -f "$out"/add.*.cpp "$out"/add.*.h
"${PYTHON:-python}" -m triton.tools.compile examples/add.py \
    --kernel-name add_kernel --signature "*u32,*u32,*u32,i32,$block" \
    --grid "(n+$block-1)/$block,1,1" --num-warps "$warps" --target vortex:32:32 \
    --out-name add --out-path "$out/add" 2>&1 | tee "$out/compile.log"
mv "$out"/add.*.h "$out/kernel.h"
mv "$out"/add.*.cpp "$out/kernel.cpp"
read -r -a config_flags <<< "$CONFIGS"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -DTTVORTEX_AOT_SHORT_NAMES \
    "${config_flags[@]}" -DBENCH_BLOCK="$block" -DBENCH_WARPS="$warps" \
    -I"$out" -I"$VORTEX_HOME/sw/runtime/include" -I"$VORTEX_BUILD_DIR/sw" \
    examples/add.cpp "$out/kernel.cpp" -L"$VORTEX_BUILD_DIR/sw/runtime" \
    -Wl,-rpath,"$VORTEX_BUILD_DIR/sw/runtime" -lvortex -o "$out/add" \
    2>&1 | tee "$out/build.log"

if (( $# == 0 )); then
    set -- 1024 1001
fi
for n in "$@"; do
    (cd "$VORTEX_BUILD_DIR"
     VORTEX_DRIVER="$driver" VORTEX_PROFILING=1 timeout 120 "$out/add" "$n" "$repeats" "$out/$driver-n$n") \
        2>&1 | tee "$out/$driver-n$n.log"
done
