"""Compare JIT/AOT add outputs and exact simulator counter deltas."""

import argparse
import csv
from pathlib import Path


def measurements(prefix):
    summary, counters = {}, {}
    with Path(str(prefix) + ".csv").open(newline="") as source:
        for row in csv.DictReader(source):
            repeat = int(row["repeat"])
            if repeat in summary:
                raise ValueError(f"{prefix}: duplicate repeat {repeat}")
            summary[repeat] = {key: int(row[key]) for key in
                               ("n", "block", "num_warps", "hw_warps", "grid", "cycles", "warp_instructions")}
    if not summary:
        raise ValueError(f"{prefix}: no measured launches")
    with Path(str(prefix) + ".counters.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["repeat"]), row["counter"])
            if key in counters:
                raise ValueError(f"{prefix}: duplicate counter {key}")
            if key[0] not in summary:
                raise ValueError(f"{prefix}: counter for unknown repeat {key[0]}")
            counters[key] = int(row["delta"])
    for repeat, row in summary.items():
        for name in ("cycles", "warp_instructions"):
            if counters.get((repeat, name)) != row[name] or (row["n"] and row[name] <= 0):
                raise ValueError(f"{prefix}: inconsistent or empty {name} at repeat {repeat}")
    return summary, counters


def compare(jit, aot):
    jit_summary, jit_counters = measurements(jit)
    aot_summary, aot_counters = measurements(aot)
    for label, left, right in (("launch", jit_summary, aot_summary), ("counter", jit_counters, aot_counters)):
        if left.keys() != right.keys():
            raise ValueError(f"{label} keys differ: JIT-only={left.keys() - right.keys()}, AOT-only={right.keys() - left.keys()}")
        differences = [(key, left[key], right[key]) for key in left if left[key] != right[key]]
        if differences:
            raise ValueError(f"{label} mismatch (key, JIT, AOT): {differences}")
    jit_output = Path(str(jit) + ".output.bin").read_bytes()
    aot_output = Path(str(aot) + ".output.bin").read_bytes()
    if not jit_output or jit_output != aot_output:
        raise ValueError("Output buffers differ, including guard elements, or are empty")
    print(f"PASS JIT/AOT agreement: {len(jit_summary)} launches, "
          f"{len(jit_counters)} exact counter deltas, {len(jit_output)} output bytes including guards")
    # Host timings measure different Python/C++ execution paths and are not deterministic.
    # cycles_per_element is derived from the checked integers and rounded in the C++ CSV.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jit_prefix", type=Path)
    parser.add_argument("aot_prefix", type=Path)
    args = parser.parse_args()
    try:
        compare(args.jit_prefix, args.aot_prefix)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"FAIL JIT/AOT agreement: {error}\n")


if __name__ == "__main__":
    main()
