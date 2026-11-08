"""Generated host ABI tests; no simulator or GPU required."""

from pathlib import Path
from importlib.resources import files
import subprocess
import tempfile
import unittest


from tt_vortex.sdk import sdk_paths

TEMPLATES = files("triton.tools.extra.vortex")
INCLUDE = sdk_paths()[0] / "sw/runtime/include"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)

    def render(self, name):
        params = dict(kernel_name=name, triton_kernel_name="example", bin_size=1,
                      bin_data="0", signature="", num_args=2,
                      arg_pointers="&global_scratch, &profile_scratch",
                      kernel_docstring="", gridX=0, gridY=1, gridZ=1,
                      warp_size=32, num_warps=4, shared=0)
        for extension in ("cpp", "h"):
            template = (TEMPLATES / f"compile.{extension}").read_text()
            (self.output / f"{name}.{extension}").write_text(template.format(**params))

    def run_command(self, *args):
        result = subprocess.run(args, cwd=self.output, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_argument_blob_abi(self):
        self.render("example_hash_")
        test = r'''
#include "example_hash_.cpp"
#include <cassert>
int main() {
    uint8_t a = 0x12, d = 0x90;
    uint16_t b = 0x3456;
    uint32_t c = 0x789abcde;
    uint64_t address = 0x1122334455667788;
    uint64_t scratch = ~uint64_t(0);
    auto blob = pack_arguments(&a, &b, &c, &d, &address, &scratch, &scratch);
    // Natural alignment: a@0, b@2, c@4, d@8, address@16; padding is zero.
    const unsigned char expected[] = {
        0x12, 0, 0x56, 0x34, 0xde, 0xbc, 0x9a, 0x78,
        0x90, 0, 0, 0, 0, 0, 0, 0,
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11
    };
    assert(blob.size == sizeof(expected));
    assert(std::memcmp(blob.data.data(), expected, sizeof(expected)) == 0);
    auto tail = pack_arguments(&a, &scratch, &scratch);
    assert(tail.size == 8 && tail.data[0] == 0x12);
    for (size_t i = 1; i < tail.size; ++i) assert(tail.data[i] == 0);
    assert(pack_arguments(&scratch, &scratch).size == 0);
    auto many = pack_arguments(MANY_ARGS &scratch, &scratch);
    assert(many.size == 160);
    for (size_t i = 0; i < many.size; i += 4)
        assert(std::memcmp(many.data.data() + i, &c, 4) == 0);
}
'''.replace("MANY_ARGS", "&c, " * 40)
        (self.output / "test.cpp").write_text(test)
        # Uncalled launcher functions are discarded, avoiding a runtime dependency.
        self.run_command("c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                         "-ffunction-sections", "-Wl,--gc-sections", "-I", str(INCLUDE),
                         "test.cpp", "-o", "test")
        self.run_command(str(self.output / "test"))

    def test_c_linkage_and_specialization_names(self):
        for name in ("example_hash1_", "example_hash2_"):
            self.render(name)
            flags = ["-DTTVORTEX_AOT_SHORT_NAMES"] if name == "example_hash1_" else []
            self.run_command("c++", "-std=c++17", "-O2", "-ffunction-sections",
                             *flags, "-I", str(INCLUDE), "-c", f"{name}.cpp")
            symbols = self.run_command("nm", "--defined-only", f"{name}.o").stdout
            self.assertIn(f" T {name}\n", symbols)
            self.assertNotIn(" T example\n", symbols)
        (self.output / "caller.c").write_text('''
#include "example_hash2_.h"
#define TTVORTEX_AOT_SHORT_NAMES
#include "example_hash1_.h"
#include "example_hash1_.h"
int main(void) {
    return example(0, 0) != VX_SUCCESS || example_hash2_(0, 0) != VX_SUCCESS;
}
''')
        self.run_command("cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                         "-I", str(INCLUDE), "-c", "caller.c")
        self.run_command("c++", "caller.o", "example_hash1_.o", "example_hash2_.o",
                         "-Wl,--gc-sections", "-o", "test")
        self.run_command(str(self.output / "test"))
        (self.output / "collision.c").write_text('''
#define TTVORTEX_AOT_SHORT_NAMES
#include "example_hash1_.h"
#include "example_hash2_.h"
''')
        result = subprocess.run(["cc", "-I", str(INCLUDE), "-fsyntax-only", "collision.c"],
                                cwd=self.output, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Short names already selected", result.stderr)


if __name__ == "__main__":
    unittest.main()
