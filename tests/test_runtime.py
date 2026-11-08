"""Host ABI checks and JIT execution with a configured simulator SDK.

Select the native runtime with VORTEX_DRIVER (defaults to simx).
"""

import ctypes as ct
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np
import triton
import triton.language as tl

from tt_vortex.compiler import sdk_paths
from tt_vortex.driver import pack_arguments
from tt_vortex.runtime import CounterSnapshot, Device, _LaunchInfo, _QueueInfo, _counter_definitions


@triton.jit
def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + offsets, offsets < n, other=0)
    tl.store(dst + offsets, value, offsets < n)


@triton.jit
def scalar_kernel(dst, value, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(dst + offsets, value)


class ABITests(unittest.TestCase):
    def test_snapshot_deltas_reject_mismatched_measurements(self):
        before = CounterSnapshot({"cycles": 12, "core.DIVERGENCE": 0}, 5, 0)
        after = CounterSnapshot({"cycles": 41, "core.DIVERGENCE": 3}, 5, 0)
        self.assertEqual(after.delta(before), {"cycles": 29, "core.DIVERGENCE": 3})
        with self.assertRaises(TypeError):
            before["cycles"] = 0
        for invalid in (CounterSnapshot(dict(before), 6, 0),
                        CounterSnapshot(dict(before), 5, 1),
                        CounterSnapshot({"cycles": 12}, 5, 0)):
            with self.assertRaises(ValueError):
                after.delta(invalid)
        with self.assertRaisesRegex(ValueError, "decreased"):
            before.delta(after)

    def test_argument_alignment_and_constexpr(self):
        signature = dict(a="u8", b="u16", c="i32", d="u8", ptr="*i32", block="constexpr")
        blob = pack_arguments(signature, (0x12, 0x3456, -2, 0x90, 0x1122334455667788, 256))
        self.assertEqual(blob, bytes.fromhex(
            "12 00 56 34 fe ff ff ff 90 00 00 00 00 00 00 00 88 77 66 55 44 33 22 11"))
        self.assertEqual(pack_arguments({"p": "*i32"}, (None,)), b"\0" * 8)
        self.assertEqual(pack_arguments({"b": "constexpr"}, (64,)), b"")
        with self.assertRaises(TypeError):
            pack_arguments({"x": "fp32"}, (1.0,))

    def test_ctypes_struct_layout_matches_c_header(self):
        home = sdk_paths()[0]
        source = r'''
#include <vortex2.h>
#include <VX_types.h>
#include <cstdio>
#include <cstddef>
int main() {
  printf("%zu %zu %zu %zu %zu %u %u %u %u", sizeof(vx_queue_info_t), sizeof(vx_launch_info_t),
         offsetof(vx_launch_info_t, args_host), offsetof(vx_launch_info_t, grid_dim),
         offsetof(vx_launch_info_t, cluster_dim), VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_DIVERGENCE,
         VX_DCR_MPM_CLASS_DCACHE, VX_CSR_MPM_DCACHE_MISS_R);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "abi.cpp").write_text(source)
            subprocess.run(["c++", "-I", str(home / "sw/runtime/include"), str(path / "abi.cpp"),
                            "-I", str(sdk_paths()[3] / "sw"),
                            "-o", str(path / "abi")], check=True)
            actual = list(map(int, subprocess.check_output([path / "abi"], text=True).split()))
        expected = [ct.sizeof(_QueueInfo), ct.sizeof(_LaunchInfo), _LaunchInfo.args_host.offset,
                    _LaunchInfo.grid_dim.offset, _LaunchInfo.cluster_dim.offset]
        definitions = _counter_definitions(home)
        expected.extend((*definitions["core.DIVERGENCE"], *definitions["dcache.MISS_R"]))
        self.assertEqual(actual, expected)


@unittest.skipUnless(os.environ.get("VORTEX_BUILD_DIR"), "Set VORTEX_BUILD_DIR to run simulator tests")
class RuntimeTests(unittest.TestCase):
    def test_named_performance_counters_for_masked_copy(self):
        with Device(driver=os.environ.get("VORTEX_DRIVER", "simx")) as dev:
            src = dev.from_numpy(np.arange(64, dtype=np.int32))
            dst = dev.full(64, -1)
            copy_kernel[(1,)](src, dst, 17, BLOCK=64, num_warps=1)
            before = dev.snapshot()
            copy_kernel[(1,)](src, dst, 17, BLOCK=64, num_warps=1)
            after = dev.snapshot()
            delta = after.delta(before)
            for name in ("cycles", "warp_instructions", "core.BRANCHES", "core.DIVERGENCE",
                         "core.LOADS", "core.STORES", "dcache.READS", "dcache.WRITES"):
                self.assertGreater(delta[name], 0, name)
            self.assertIn("core.STALL_SCRB", delta)
            self.assertIn("icache.MISS_R", delta)
            self.assertEqual(dev.counter("core.DIVERGENCE"), after["core.DIVERGENCE"])
            self.assertEqual(set(dev.snapshot(("cycles", "core.DIVERGENCE"))), {"cycles", "core.DIVERGENCE"})
            with self.assertRaises(KeyError):
                dev.snapshot(("missing.counter",))
            with self.assertRaises(ValueError):
                dev.counter("cycles", core=dev.properties["num_cores"])

    def test_jit_copy_masked_tail_multiple_ctas_and_specializations(self):
        for iteration in range(2):  # A new device must not reuse an unloaded module.
            with Device(driver=os.environ.get("VORTEX_DRIVER", "simx")) as dev:
                original = np.arange(300, dtype=np.int32) * -17
                src = dev.from_numpy(original)
                dst = dev.full(300, -999)
                for n, block, warps in ((0, 64, 1), (1, 64, 1), (17, 64, 1), (257, 128, 4), (17, 64, 1)):
                    dst.fill(-999)
                    copy_kernel[(triton.cdiv(n, block),)](src, dst, n, BLOCK=block, num_warps=warps)
                    actual = dst.numpy()
                    np.testing.assert_array_equal(actual[:n], original[:n])
                    self.assertTrue(np.all(actual[n:] == -999))
                with self.assertRaises(ValueError):
                    src.copy_from(np.zeros(1, np.int32))
                with self.assertRaises(TypeError):
                    dev.empty(1, dtype=object)
                src.close()
                with self.assertRaises(RuntimeError):
                    src.data_ptr()
            with self.assertRaises(RuntimeError):
                dst.numpy()

    def test_integer_scalar_argument_and_cached_launch(self):
        with Device(driver=os.environ.get("VORTEX_DRIVER", "simx")) as dev:
            dst = dev.empty(32, np.int32)
            for value in (-7, 42):
                scalar_kernel[(1,)](dst, value, BLOCK=32, num_warps=1)
                np.testing.assert_array_equal(dst.numpy(), np.full(32, value, np.int32))
            snapshot = dev.snapshot()
            self.assertGreater(snapshot["cycles"], 0)
            self.assertGreater(snapshot["warp_instructions"], 0)


if __name__ == "__main__":
    unittest.main()
