#!/usr/bin/env bash
# Exercise the standard Triton AOT CLI, retaining diagnostics and completed IR stages.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

mkdir -p "${BASIC_OUT_DIR:-build/basic}"
out=$(cd "${BASIC_OUT_DIR:-build/basic}" && pwd)
export TRITON_DEFAULT_BACKEND=vortex TRITON_STORE_BINARY_ONLY=0
export TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1
printf 'kernel,exit_code\n' > "$out/results.csv"
if (( $# == 0 )); then
    set -- multiply gather float_add block_sum prefix_sum atomic_sum transpose
fi
for kernel in "$@"; do
    grid='(n+255)/256,1,1'
    case "$kernel" in
        multiply|gather) signature='*i32,*i32,*i32,i32,256' ;;
        float_add) signature='*fp32,*fp32,*fp32,i32,256' ;;
        block_sum|prefix_sum|atomic_sum) signature='*i32,*i32,i32,256' ;;
        transpose)
            signature='*i32,*i32,i32,i32,16,16'
            grid='(cols+15)/16,(rows+15)/16,1' ;;
        *) echo "Unknown kernel: $kernel" >&2; exit 2 ;;
    esac
    directory="$out/$kernel"
    mkdir -p "$directory"
    # Replace generated artifacts so a later failure cannot leave an old VXBIN.
    rm -rf "$directory/ir"
    rm -f "$directory/$kernel".{ttir,ttgir,llir,vxbin} "$directory/$kernel".*.{cpp,h}
    status=0
    TRITON_CACHE_DIR="$directory/cache" TRITON_DUMP_DIR="$directory/ir" \
        "${PYTHON:-python}" -m triton.tools.compile examples/basic.py \
        --kernel-name "$kernel" --signature "$signature" --grid "$grid" \
        --num-warps 4 --target vortex:32:32 --out-name "$kernel" \
        --out-path "$directory/$kernel" > "$directory/compile.log" 2>&1 || status=$?
    for artifact in "$directory"/ir/*/"$kernel".{ttir,ttgir,llir,vxbin}; do
        cp "$artifact" "$directory/"
    done
    printf '%s,%s\n' "$kernel" "$status" >> "$out/results.csv"
    printf '%-12s exit=%s  %s\n' "$kernel" "$status" "$directory/compile.log"
done
# Failure is an observation here, not an assertion of what a backend should support.
echo "Compiler outcomes: $out/results.csv"
