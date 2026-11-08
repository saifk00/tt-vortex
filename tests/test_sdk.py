"""SDK discovery must work independently of checkout and install locations."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from tt_vortex.sdk import sdk_paths


class SDKTests(unittest.TestCase):
    def test_missing_sdk_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "VORTEX_SDK_PATH"):
                sdk_paths()

    def test_sdk_root_and_component_overrides(self):
        with patch.dict(os.environ, {"VORTEX_SDK_PATH": "/opt/vortex"}, clear=True):
            self.assertEqual(sdk_paths(), tuple(map(Path, (
                "/opt/vortex/source", "/opt/vortex/tools",
                "/opt/vortex/tools/llvm-vortex", "/opt/vortex/build"))))
            with patch.dict(os.environ, {"VORTEX_HOME": "/src/vortex",
                                        "VORTEX_BUILD_DIR": "/tmp/rtl",
                                        "VORTEX_TOOLDIR": "/opt/toolchains",
                                        "VORTEX_LLVM_PATH": "/opt/custom-llvm"}):
                self.assertEqual(sdk_paths(), tuple(map(Path, (
                    "/src/vortex", "/opt/toolchains", "/opt/custom-llvm", "/tmp/rtl"))))

    def test_existing_source_sdk(self):
        with patch.dict(os.environ, {"VORTEX_HOME": "/src/vortex"}, clear=True):
            self.assertEqual(sdk_paths(), tuple(map(Path, (
                "/src/vortex", "/src/vortex/tools",
                "/src/vortex/tools/llvm-vortex", "/src/vortex/build32"))))
