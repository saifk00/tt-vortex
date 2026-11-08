"""Native MLIR driver tests; no Python backend import or Vortex SDK required."""

import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
OPT = Path(os.environ.get("TTVORTEX_OPT", ROOT / "build/bin/ttvortex-opt"))
IR = Path(__file__).parent / "ir"
TO_GPU = "convert-triton-to-tritongpu{target=vortex:32 num-warps=4 threads-per-warp=32 num-ctas=1},canonicalize,cse"
TO_LLVM = "convert-scf-to-cf,convert-triton-gpu-to-vortex-llvm,reconcile-unrealized-casts,canonicalize,cse,symbol-dce"


@unittest.skipUnless(OPT.is_file(), "Build the native driver with make opt")
class OptimizerTests(unittest.TestCase):
    def run_opt(self, *args, input=None):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("VORTEX_", "TRITON_")) and key != "PYTHONPATH"}
        return subprocess.run([str(OPT), *args], input=input, text=True,
                              capture_output=True, env=env)

    def test_ttir_to_ttgir_to_llvm(self):
        ttgir = self.run_opt(str(IR / "square.ttir"), f"--pass-pipeline=builtin.module({TO_GPU})")
        self.assertEqual(ttgir.returncode, 0, ttgir.stderr)
        self.assertIn('ttg.target = "vortex:32"', ttgir.stdout)
        self.assertIn("threadsPerWarp = [32]", ttgir.stdout)
        lowered = self.run_opt(f"--pass-pipeline=builtin.module({TO_LLVM})", input=ttgir.stdout)
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        self.assertIn("llvm.func @square", lowered.stdout)
        self.assertIn("llvm.mul", lowered.stdout)
        self.assertNotIn("tt.load", lowered.stdout)
        combined = self.run_opt(str(IR / "square.ttir"),
                                f"--pass-pipeline=builtin.module({TO_GPU},{TO_LLVM})")
        self.assertEqual(combined.returncode, 0, combined.stderr)
        self.assertEqual(combined.stdout, lowered.stdout)

    def test_block_sum_lowering(self):
        result = self.run_opt(str(IR / "block_sum.ttgir"),
                              f"--pass-pipeline=builtin.module({TO_LLVM})")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("llvm.func @block_sum", result.stdout)
        self.assertIn("ttg.shared = 16", result.stdout)
        self.assertIn(".insn r 0x0b, 6, 1", result.stdout)
        self.assertIn(".insn r 0x0b, 4, 0", result.stdout)
        self.assertNotIn("tt.reduce", result.stdout)
        self.assertNotIn("global_smem", result.stdout)

    def test_bytecode_round_trip(self):
        encoded = subprocess.run([str(OPT), str(IR / "square.ttir"), "--emit-bytecode"],
                                 capture_output=True)
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        decoded = subprocess.run([str(OPT)], input=encoded.stdout, capture_output=True)
        self.assertEqual(decoded.returncode, 0, decoded.stderr)
        self.assertIn(b"tt.func public @square", decoded.stdout)
