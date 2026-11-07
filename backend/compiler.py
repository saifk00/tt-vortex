"""Native Triton lowering and Vortex device image compilation."""

from dataclasses import dataclass, asdict
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from triton import knobs
from triton._C.libtriton import ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language


@lru_cache(maxsize=1)
def native():
    # Delay loading until compilation: Python backend discovery also runs at setup.
    try:
        from . import _C
    except ImportError as error:
        raise RuntimeError("Build the Vortex extension first: make compiler (workspace) or make (tt-vortex)") from error
    passes.plugin.extend_with(_C.__file__)
    return _C


def sdk_paths():
    home = Path(os.environ.get("VORTEX_HOME", Path(__file__).resolve().parents[2] / "vortex")).resolve()
    tools = Path(os.environ.get("VORTEX_TOOLDIR", home / "tools")).resolve()
    llvm_dir = Path(os.environ.get("VORTEX_LLVM_PATH", tools / "llvm-vortex")).resolve()
    build = Path(os.environ.get("VORTEX_BUILD_DIR", home / "build32")).resolve()
    return home, tools, llvm_dir, build


@dataclass(frozen=True)
class VortexOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 32
    debug: bool = False
    sanitize_overflow: bool = False
    enable_fp_fusion: bool = False
    backend_name: str = "vortex"
    supported_fp8_dtypes: tuple = ()
    allowed_dot_input_precisions: tuple = ()
    max_num_imprecise_acc_default: int = 0

    def __post_init__(self):
        if self.num_ctas != 1:
            raise ValueError("Vortex currently supports one CTA per program")
        if self.num_warps < 1 or self.num_warps & (self.num_warps - 1):
            raise ValueError("num_warps must be a positive power of two")
        if self.warp_size < 1 or self.warp_size > 32 or self.warp_size & (self.warp_size - 1):
            raise ValueError("Vortex warp_size must be a power of two between 1 and 32")

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class VortexBackend(BaseBackend):
    binary_ext = "vxbin"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "vortex"

    def __init__(self, target):
        try:
            arch = int(target.arch)
        except (TypeError, ValueError):
            raise ValueError(f"Vortex backend expects a numeric arch, got {target.arch!r}") from None
        try:
            warp_size = int(target.warp_size)
        except (TypeError, ValueError):
            raise ValueError(f"Vortex backend expects a numeric warp_size, got {target.warp_size!r}") from None
        target = GPUTarget(target.backend, arch, warp_size)
        super().__init__(target)
        if target.arch != 32:
            raise ValueError("The initial Vortex backend supports RV32 only")
        if knobs.compilation.store_binary_only:
            raise ValueError("Use TRITON_STORE_BINARY_ONLY=0: upstream Triton's binary-only cache only retains CUDA/HIP artifacts")

    def hash(self):
        home, tools, llvm_dir, build = sdk_paths()
        digest = hashlib.sha256(Path(__file__).read_bytes())
        digest.update(str(tools).encode())
        digest.update(Path(native().__file__).read_bytes())
        for path in (home / "sw/kernel/src/vx_start.S", home / "sw/kernel/src/common.h",
                     home / "sw/kernel/scripts/link32.ld", home / "sw/kernel/scripts/vxbin.py",
                     build / "sw/VX_config.h", build / "sw/VX_types.h",
                     llvm_dir / "bin/clang", llvm_dir / "bin/llvm-objcopy"):
            digest.update(str(path).encode())
            if path.is_file():
                if path.parent == llvm_dir / "bin":
                    stat = path.stat()
                    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
                else:
                    digest.update(path.read_bytes())
        return f"vortex-rv32-{self.target.warp_size}-vxbin-v2-{digest.hexdigest()}"

    def parse_options(self, options):
        values = {key: value for key, value in options.items() if key in VortexOptions.__dataclass_fields__}
        values["warp_size"] = self.target.warp_size
        return VortexOptions(**values)

    def pack_metadata(self, metadata):
        return metadata.num_warps, metadata.num_ctas, metadata.warp_size

    def get_codegen_implementation(self, options):
        return {}

    def get_module_map(self):
        return {}

    def load_dialects(self, context):
        # Vortex currently introduces no new dialect; its patterns emit LLVM.
        ir.load_dialects(context)

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, "vortex:32", options.num_warps, options.warp_size, options.num_ctas)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttgir")
        return mod

    @staticmethod
    def make_llir(mod, metadata, options):
        vortex = native()
        metadata["name"] = mod.get_entry_func_name()
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.plugin.add_vortex_to_llvm(pm)
        passes.convert.add_reconcile_unrealized_casts(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_llir")
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        vortex.set_target(llvm_mod)
        metadata["args_size"] = vortex.add_launch_wrapper(llvm_mod, metadata["name"])
        metadata.update(shared=0, global_scratch_size=0, global_scratch_align=1,
                        profile_scratch_size=0, profile_scratch_align=1)
        return str(llvm_mod)

    @staticmethod
    def make_vxbin(src, metadata, options):
        home, tools, llvm_dir, build = sdk_paths()
        gcc = tools / "riscv32-gnu-toolchain"
        clang = llvm_dir / "bin/clang"
        for path in (clang, build / "sw/VX_config.h", home / "sw/kernel/src/vx_start.S"):
            if not path.is_file():
                raise RuntimeError(f"Missing Vortex SDK file: {path}. Set VORTEX_HOME, VORTEX_TOOLDIR and VORTEX_BUILD_DIR to an RV32 SDK/build.")
        common = [str(clang), "--target=riscv32-unknown-elf", "-march=rv32imaf", "-mabi=ilp32f",
                  f"--sysroot={gcc}/riscv32-unknown-elf", f"--gcc-toolchain={gcc}",
                  "-Xclang", "-target-feature", "-Xclang", "+xvortex",
                  "-Xclang", "-target-feature", "-Xclang", "+zicond",
                  "-mllvm", "-disable-loop-idiom-all", "-O1", "-mcmodel=medany",
                  "-fno-vectorize", "-fno-slp-vectorize"]

        def run(command, **kwargs):
            result = subprocess.run([str(arg) for arg in command], capture_output=True, text=True, **kwargs)
            if result.returncode:
                raise RuntimeError(f"Vortex tool failed: {' '.join(map(str, command))}\n{result.stdout}{result.stderr}")

        with tempfile.TemporaryDirectory(prefix="ttvortex-") as tmp:
            tmp = Path(tmp)
            (tmp / "kernel.ll").write_text(src)
            run([*common, "-c", tmp / "kernel.ll", "-o", tmp / "kernel.o"])
            run([*common, "-D__VORTEX__", "-DKMU_ENABLE", "-I", home / "sw/kernel/include",
                 "-I", build / "sw", "-c", home / "sw/kernel/src/vx_start.S", "-o", tmp / "startup.o"])
            run([*common, "-nostdlib", "-nostartfiles", tmp / "startup.o", tmp / "kernel.o",
                 f"-Wl,-Bstatic,--gc-sections,-T,{home}/sw/kernel/scripts/link32.ld,--defsym=STARTUP_ADDR=0x80000000",
                 "-o", tmp / "kernel.elf"])
            env = dict(os.environ, OBJCOPY=str(llvm_dir / "bin/llvm-objcopy"))
            run([sys.executable, home / "sw/kernel/scripts/vxbin.py", tmp / "kernel.elf", tmp / "kernel.vxbin"], env=env)
            return (tmp / "kernel.vxbin").read_bytes()

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("The initial Vortex backend accepts Triton kernels, not Gluon")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["vxbin"] = lambda src, metadata: self.make_vxbin(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, self.target.arch)
