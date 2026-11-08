"""The backend must be importable before Triton, including its public runtime."""

import subprocess
import sys
import unittest


class PackageTests(unittest.TestCase):
    def test_import_order(self):
        for module in ("tt_vortex.compiler", "tt_vortex.driver", "tt_vortex.runtime", "triton"):
            with self.subTest(first_import=module):
                result = subprocess.run([
                    sys.executable, "-c",
                    f"import {module}; from tt_vortex.compiler import native; native()",
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
