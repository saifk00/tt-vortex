"""CPU-only regressions through Triton's normal compiler API."""

from contextlib import ExitStack
import inspect
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import triton
import triton.language as tl
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend
from triton.backends.vortex.compiler import sdk_paths


@triton.jit
def vortex_copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + offsets, mask=offsets < n, other=0)
    tl.store(dst + offsets, values, mask=offsets < n)


def source():
    return ASTSource(vortex_copy_kernel,
                     signature={"src": "*u16", "dst": "*u16", "n": "i32", "BLOCK": "constexpr"},
                     constexprs={"BLOCK": 1024})


class ExternalBackendTests(unittest.TestCase):
    def setUp(self):
        scope = ExitStack()
        self.addCleanup(scope.close)
        self.output = Path(scope.enter_context(tempfile.TemporaryDirectory()))
        scope.enter_context(knobs.cache.scope())
        scope.enter_context(knobs.compilation.scope())
        knobs.cache.dir = str(self.output)
        knobs.compilation.store_binary_only = False

    def test_backend_is_external(self):
        backend = make_backend(GPUTarget("vortex", 32, 32))
        self.assertEqual(type(backend).__name__, "VortexBackend")
        expected = Path(__file__).resolve().parents[1] / "backend/compiler.py"
        self.assertEqual(Path(inspect.getfile(type(backend))).resolve(), expected)

    def test_string_target_matches_integer_target(self):
        target = GPUTarget(*"vortex:32:32".split(":"))
        backend = make_backend(target)
        self.assertEqual(backend.target, GPUTarget("vortex", 32, 32))
        self.assertIs(type(backend.target.arch), int)
        self.assertIs(type(backend.target.warp_size), int)
        knobs.compilation.always_compile = True
        actual = triton.compile(source(), target=target)
        expected = triton.compile(source(), target=GPUTarget("vortex", 32, 32))
        self.assertEqual(actual.asm["llir"], expected.asm["llir"])

    def test_nonnumeric_target_is_rejected(self):
        for target, field in ((GPUTarget("vortex", "rv32", "32"), "arch"),
                              (GPUTarget("vortex", "32", "thirty-two"), "warp_size")):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"numeric {field}"):
                    make_backend(target)

    def test_copy(self):
        for warp in (4, 32):
            with self.subTest(warp_size=warp):
                kernel = triton.compile(source(), target=GPUTarget("vortex", 32, warp), options={"num_warps": 4})
                self.assertIn(f"threadsPerWarp = [{warp}]", kernel.asm["ttgir"])
                llir = kernel.asm["llir"]
                self.assertIn('target triple = "riscv32-unknown-elf"', llir)
                self.assertIn('asm "csrr $0, $1", "=r,i"(i32 3286)', llir)
                self.assertIn('asm "csrr $0, $1", "=r,i"(i32 3283)', llir)
                self.assertNotIn("llvm.nvvm", llir)
                self.assertNotIn("llvm.amdgcn", llir)
                self.assertEqual(llir.count("load i16"), 1024 // (warp * 4))
                self.assertEqual(llir.count("store i16"), 1024 // (warp * 4))
                self.assertIn("br i1", llir)
                self.assertIn("define void @kernel_main(ptr", llir)
                self.assertEqual(kernel.metadata.args_size, 24)
                self.assertEqual(struct.unpack_from("<Q", kernel.asm["vxbin"])[0], 0x80000000)
                self.assertTrue(kernel.asm["vxbin"].endswith(b"VXSYMTAB"))

    def test_standard_aot_cli(self):
        env = dict(os.environ, TRITON_DEFAULT_BACKEND="vortex", TRITON_CACHE_DIR=str(self.output / "cache"))
        result = subprocess.run([
            sys.executable, "-m", "triton.tools.compile", str(Path(__file__).resolve()),
            "--kernel-name", "vortex_copy_kernel", "--signature", "*u16,*u16,i32,1024",
            "--grid", "(n+1023)/1024,1,1", "--out-name", "vortex_tmp", "--target", "vortex:32:32",
        ], cwd=self.output, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launcher, = self.output.glob("vortex_tmp.*.cpp")
        header, = self.output.glob("vortex_tmp.*.h")
        self.assertIn("vx_module_load_bytes", launcher.read_text())
        self.assertIn("vx_enqueue_launch", launcher.read_text())
        self.assertIn("uint64_t src, uint64_t dst, int32_t n", header.read_text())
        subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-c", str(launcher),
                        "-I", str(sdk_paths()[0] / "sw/runtime/include"), "-o", str(self.output / "launcher.o")], check=True)

    def test_aot_python_api_uses_vortex_types(self):
        from triton.backends.vortex.driver import VortexDriver
        from triton.tools.compile import CompileArgs, compile_kernel

        with patch.object(triton.runtime.driver, "_active", VortexDriver()):
            _, files = compile_kernel(CompileArgs(
                path=str(Path(__file__).resolve()), kernel_name="vortex_copy_kernel",
                signature="*u16,*u16,i32,1024", grid="(n+1023)/1024,1,1",
                target="vortex:32:32", out_name="vortex_api", out_path=self.output / "vortex_api"))
        launcher = next(path for path in files if path.suffix == ".cpp")
        self.assertIn("uint64_t src, uint64_t dst", launcher.read_text())
        for path in files:
            for foreign_type in ("CUDA_VERSION", "CUdeviceptr", "hipDeviceptr_t"):
                self.assertNotIn(foreign_type, path.read_text())
        subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-c", str(launcher),
                        "-I", str(sdk_paths()[0] / "sw/runtime/include"), "-o", str(self.output / "vortex_api.o")], check=True)

    def test_saved_ttgir_layouts(self):
        original = triton.compile(source(), target=GPUTarget("vortex", 32, 32))
        for size in (1, 8):
            with self.subTest(size_per_thread=size):
                path = self.output / f"layout-{size}.ttgir"
                path.write_text(original.asm["ttgir"].replace("sizePerThread = [1]", f"sizePerThread = [{size}]"))
                result = triton.compile(str(path), target=GPUTarget("vortex", 32, 32))
                self.assertEqual(result.asm["llir"].count("load i16"), 8)
                self.assertEqual(result.asm["llir"].count("store i16"), 8)

    def test_binary_only_reports_upstream_limitation(self):
        knobs.compilation.store_binary_only = True
        with self.assertRaisesRegex(ValueError, "TRITON_STORE_BINARY_ONLY=0"):
            make_backend(GPUTarget("vortex", 32, 32))

    def test_multi_cta_is_rejected(self):
        backend = make_backend(GPUTarget("vortex", 32, 32))
        with self.assertRaisesRegex(ValueError, "one CTA per program"):
            backend.parse_options({"num_ctas": 2})


if __name__ == "__main__":
    unittest.main()
