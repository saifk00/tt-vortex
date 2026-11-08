"""JIT vector addition on Vortex; run with a configured SDK and SimX runtime."""

import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import time
import numpy as np
import triton
import triton.language as tl
from tt_vortex.runtime import Device


@triton.jit
def add_kernel(a, b, dst, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + offsets, mask=offsets < n, other=0)
    y = tl.load(b + offsets, mask=offsets < n, other=0)
    tl.store(dst + offsets, x + y, mask=offsets < n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--driver", choices=("simx", "rtlsim", "firesim"), default="simx")
    parser.add_argument("--output-prefix", type=Path, help="Write summary/counter CSVs and the output buffer")
    args = parser.parse_args()
    if args.n < 0 or args.repeats < 1 or args.block < 1 or args.block & (args.block - 1):
        parser.error("n must be nonnegative, repeats positive, and block a positive power of two")
    with ExitStack() as resources:
        device = resources.enter_context(Device(driver=args.driver))
        line = device.properties["cache_line_size"]
        capacity = triton.cdiv(args.n * 4 + line, line) * line // 4
        i = np.arange(capacity, dtype=np.uint32)
        a_host = i * np.uint32(0x1020305) + np.uint32(0x89ABCDEF)
        b_host = (i * 17) ^ np.uint32(0x7FFF0000)
        summary, raw = None, None
        if args.output_prefix is not None:
            args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
            summary = csv.writer(resources.enter_context(Path(str(args.output_prefix) + ".csv").open("w", newline="")))
            raw = csv.writer(resources.enter_context(Path(str(args.output_prefix) + ".counters.csv").open("w", newline="")))
            summary.writerow(("repeat", "n", "block", "num_warps", "hw_warps", "grid", "cycles",
                              "warp_instructions", "cycles_per_element", "host_launch_ms"))
            raw.writerow(("repeat", "counter", "delta"))
        a, b, dst = (device.empty(capacity, np.uint32) for _ in range(3))

        def prepare():
            a.copy_from(a_host)
            b.copy_from(b_host)
            dst.fill(0xDEADBEEF)

        grid = (triton.cdiv(args.n, args.block),)
        # Warm up compilation, loading, and execution outside the measurement.
        prepare()
        add_kernel[grid](a, b, dst, args.n, BLOCK=args.block, num_warps=args.warps)
        device.synchronize()
        print(f"CONFIG driver={args.driver} device={device.properties} block={args.block} warps={args.warps}")
        for repeat in range(args.repeats):
            prepare()
            before = device.snapshot()
            start = time.perf_counter()
            add_kernel[grid](a, b, dst, args.n, BLOCK=args.block, num_warps=args.warps)
            device.synchronize()
            host_ms = (time.perf_counter() - start) * 1000
            after = device.snapshot()
            result = dst.numpy()
            np.testing.assert_array_equal(result[:args.n], (a_host + b_host)[:args.n])
            np.testing.assert_array_equal(result[args.n:], np.uint32(0xDEADBEEF))
            delta = after.delta(before)
            cycles, instructions = delta["cycles"], delta["warp_instructions"]
            if args.n and (cycles <= 0 or instructions <= 0):
                raise RuntimeError("Measured launch returned empty cycle/instruction counters")
            print(f"PASS add n={args.n} repeat={repeat} cycles={cycles} warp_instructions={instructions} "
                  f"host_launch_ms={host_ms:.3f}")
            for name, value in delta.items():
                print(f"COUNTER repeat={repeat} {name}={value}")
                if raw is not None:
                    raw.writerow((repeat, name, value))
            if summary is not None:
                summary.writerow((repeat, args.n, args.block, args.warps, device.properties["num_warps"],
                                  grid[0], cycles, instructions, cycles / args.n if args.n else 0, host_ms))
        if args.output_prefix is not None:
            result.tofile(str(args.output_prefix) + ".output.bin")
            print(f"FILES {args.output_prefix}.csv {args.output_prefix}.counters.csv {args.output_prefix}.output.bin")


if __name__ == "__main__":
    main()
