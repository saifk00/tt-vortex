"""Numerical checks of the basic examples on a configured simulator."""

import os
import unittest

import numpy as np
import triton
import triton.language as tl

from examples.basic import atomic_sum, float_add, prefix_sum, transpose
from tt_vortex.runtime import Device


@triton.jit
def atomic_tickets(out, tickets, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    old = tl.atomic_add(out + tl.full((BLOCK,), 0, tl.int32), 1,
                        mask=i < n, sem="relaxed")
    tl.store(tickets + i, old, i < n)


@triton.jit
def scalar_atomic_ticket(out, tickets, BLOCK: tl.constexpr):
    old = tl.atomic_add(out, 1, sem="relaxed")
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(tickets + i, old)


@unittest.skipUnless(os.environ.get("VORTEX_BUILD_DIR") or os.environ.get("VORTEX_SDK_PATH"),
                     "Configure a Vortex SDK to run simulator tests")
class BasicRuntimeTests(unittest.TestCase):
    def test_transpose_layouts_tails_and_scratch_reuse(self):
        shapes = [(16, 16), (17, 29), (33, 65), (1, 37), (37, 1), (7, 11),
                  (32, 48), (65, 33), (0, 17), (17, 0)]
        tiles = [(8, 8), (8, 32), (16, 16), (32, 8), (32, 32)]
        with Device(driver=os.environ.get("VORTEX_DRIVER", "simx")) as dev:
            for dtype in (np.int32, np.float32):
                for rows, cols in shapes:
                    n = rows * cols
                    bits = np.random.default_rng(rows * 101 + cols).integers(
                        0, 1 << 32, n, dtype=np.uint32)
                    special = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                                        0x7fc12345, 0x7f812345, 1, 0x807fffff],
                                       dtype=np.uint32)
                    bits[:min(n, special.size)] = special[:min(n, special.size)]
                    original = bits.view(dtype)
                    src = dev.from_numpy(original if n else np.zeros(1, dtype=dtype))
                    dst = dev.full(n + 32, -999, dtype=dtype)
                    expected = bits.reshape(rows, cols).T.ravel()
                    for block_m, block_n in tiles:
                        for warps in (1, 2, 4):
                            with self.subTest(dtype=dtype, shape=(rows, cols),
                                              tile=(block_m, block_n), warps=warps):
                                # Include fully masked CTAs on both grid axes.
                                grid = (triton.cdiv(cols, block_n) + 1,
                                        triton.cdiv(rows, block_m) + 1)
                                for repeat in range(2):
                                    dst.fill(-999)
                                    transpose[grid](src, dst, rows, cols,
                                                    BLOCK_M=block_m, BLOCK_N=block_n,
                                                    num_warps=warps)
                                    actual = dst.numpy()
                                    # Transpose is a bit-preserving move, even
                                    # for floating-point NaNs and signed zeros.
                                    np.testing.assert_array_equal(
                                        actual[:n].view(np.uint32), expected)
                                    np.testing.assert_array_equal(actual[n:], -999)
                    np.testing.assert_array_equal(src.numpy().view(np.uint32),
                                                  bits if n else [0])
                    dst.close()
                    src.close()

    def test_float_add_masked_tails_and_ieee_values(self):
        rng = np.random.default_rng(19)
        a = rng.integers(0, 1 << 32, 777, dtype=np.uint32).view(np.float32)
        b = rng.integers(0, 1 << 32, 777, dtype=np.uint32).view(np.float32)
        info = np.finfo(np.float32)
        a[:10] = [0., -0., np.inf, -np.inf, np.nan, info.smallest_subnormal,
                  1., 1., info.max, info.tiny]
        b[:10] = [-0., -0., -np.inf, np.inf, 1., info.smallest_subnormal,
                  info.eps / 2, info.eps * 1.5, info.max, -info.tiny]
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            sums = a + b
        cases = [(256, warps, n) for warps in (1, 4)
                 for n in (0, 1, 17, 33, 127, 128, 129, 255, 256, 257, 777)]
        cases.append((16, 4, 37))
        with Device(driver=os.environ.get("VORTEX_DRIVER", "simx")) as dev:
            src_a, src_b = dev.from_numpy(a), dev.from_numpy(b)
            dst = dev.full(800, -999, dtype=np.float32)
            for block, warps, n in cases:
                with self.subTest(block=block, warps=warps, n=n):
                    dst.fill(-999)
                    float_add[(triton.cdiv(n, block) + 1,)](
                        src_a, src_b, dst, n, BLOCK=block, num_warps=warps)
                    expected = np.full(800, -999, dtype=np.float32)
                    expected[:n] = sums[:n]
                    actual = dst.numpy()
                    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                    # Compare exact rounding and signed zeros; NaN payloads
                    # may differ between host and RISC-V's canonical NaNs.
                    defined = ~np.isnan(expected)
                    np.testing.assert_array_equal(actual.view(np.uint32)[defined],
                                                  expected.view(np.uint32)[defined])

    def test_prefix_and_atomic_sums(self):
        # Mixed signs and overflow exercise i32 wraparound, not just small sums.
        original = np.random.default_rng(7).integers(
            -(1 << 31), 1 << 31, 777, dtype=np.int32)
        cases = [(256, warps, n) for warps in (1, 4)
                 for n in (0, 1, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 777)]
        cases += [(16, 4, 37), (64, 1, 129)]
        with Device(driver=os.environ.get("VORTEX_DRIVER", "simx")) as dev:
            src = dev.from_numpy(original)
            dst = dev.full(800, -999)
            total = dev.full(3, -999)
            for block, warps, n in cases:
                with self.subTest(block=block, warps=warps, n=n):
                    # Include fully masked CTAs, even when n=0; reuse scratch.
                    grid = (triton.cdiv(n, block) + 1,)
                    dst.fill(-999)
                    prefix_sum[grid](src, dst, n, BLOCK=block, num_warps=warps)
                    expected = np.full(800, -999, dtype=np.int32)
                    for start in range(0, n, block):
                        end = min(start + block, n)
                        expected[start:end] = np.cumsum(original[start:end], dtype=np.int32)
                    np.testing.assert_array_equal(dst.numpy(), expected)

                    # Guard words catch writes outside the single accumulator.
                    total.copy_from(np.array([37, -999, -999], dtype=np.int32))
                    for repeat in (1, 2):
                        atomic_sum[grid](src, total, n, BLOCK=block, num_warps=warps)
                        value = np.array(37 + repeat * original[:n].sum(dtype=np.int64)).astype(np.int32)
                        np.testing.assert_array_equal(total.numpy(), [value, -999, -999])

    def test_atomic_return_values_and_duplicate_layout_owners(self):
        with Device(driver=os.environ.get("VORTEX_DRIVER", "simx")) as dev:
            total = dev.full(1, 0)
            tickets = dev.full(800, -999)
            # Small tiles have duplicate lane/warp owners; large tiles have
            # multiple registers per thread. Every logical element updates once.
            for block, warps, n in ((16, 4, 37), (64, 4, 129), (256, 1, 777), (256, 4, 777)):
                with self.subTest(block=block, warps=warps):
                    total.fill(0)
                    tickets.fill(-999)
                    atomic_tickets[(triton.cdiv(n, block) + 1,)](
                        total, tickets, n, BLOCK=block, num_warps=warps)
                    self.assertEqual(total.numpy()[0], n)
                    actual = tickets.numpy()
                    np.testing.assert_array_equal(np.sort(actual[:n]), np.arange(n))
                    np.testing.assert_array_equal(actual[n:], -999)

            # A scalar atomic executes once per CTA and broadcasts its result.
            total.fill(0)
            tickets.fill(-999)
            scalar_atomic_ticket[(3,)](total, tickets, BLOCK=256, num_warps=4)
            self.assertEqual(total.numpy()[0], 3)
            rows = tickets.numpy()[:768].reshape(3, 256)
            np.testing.assert_array_equal(rows, np.broadcast_to(rows[:, :1], rows.shape))
            np.testing.assert_array_equal(np.sort(rows[:, 0]), [0, 1, 2])
            np.testing.assert_array_equal(tickets.numpy()[768:], -999)


if __name__ == "__main__":
    unittest.main()
