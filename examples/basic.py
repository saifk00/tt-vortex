"""Small backend coverage examples; compile with examples/compile_basic.sh.

Each program handles one tile. Sum and prefix-sum operate within that tile;
atomic_sum accumulates into a single output word that the host must zero first.
Float-add, scan, and atomic-add checks live in tests/test_basic_runtime.py.
"""

import triton
import triton.language as tl


@triton.jit
def multiply(a, b, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + i, i < n, other=0)
    y = tl.load(b + i, i < n, other=0)
    tl.store(out + i, x * y, i < n)


@triton.jit
def gather(src, indices, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(indices + i, i < n, other=0)
    # The host supplies indices in [0, source length).
    value = tl.load(src + index, i < n, other=0)
    tl.store(out + i, value, i < n)


@triton.jit
def float_add(a, b, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + i, i < n, other=0)
    y = tl.load(b + i, i < n, other=0)
    tl.store(out + i, x + y, i < n)


@triton.jit
def block_sum(src, out, n, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    i = program * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + i, i < n, other=0)
    total = tl.sum(value, axis=0)
    tl.store(out + program, total)


@triton.jit
def prefix_sum(src, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + i, i < n, other=0)
    prefix = tl.cumsum(value, axis=0)
    tl.store(out + i, prefix, i < n)


@triton.jit
def atomic_sum(src, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + i, i < n, other=0)
    pointers = out + tl.full((BLOCK,), 0, tl.int32)
    tl.atomic_add(pointers, value, mask=i < n, sem="relaxed")


@triton.jit
def transpose(src, out, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tile = tl.load(src + row[:, None] * cols + col[None, :],
                   (row[:, None] < rows) & (col[None, :] < cols), other=0)
    tl.store(out + col[:, None] * rows + row[None, :], tl.trans(tile),
             (col[:, None] < cols) & (row[None, :] < rows))
