"""Integer vector addition, compiled with python -m triton.tools.compile."""

import triton
import triton.language as tl


@triton.jit
def add_kernel(a, b, dst, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + offsets, mask=offsets < n, other=0)
    y = tl.load(b + offsets, mask=offsets < n, other=0)
    tl.store(dst + offsets, x + y, mask=offsets < n)
