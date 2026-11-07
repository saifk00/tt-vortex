"""Backend discovery descriptor for explicit-target, compile-only Vortex use."""

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
import os


class VortexDriver(DriverBase):
    @classmethod
    def is_active(cls):
        # Never replace a CUDA/HIP execution driver merely because Vortex is installed.
        return os.environ.get("TRITON_DEFAULT_BACKEND") == "vortex"

    def get_current_target(self):
        return GPUTarget("vortex", 32, 32)

    def map_python_to_cpp_type(self, ty):
        if ty.startswith("*"):
            return "uint64_t"
        return {f"{sign}{bits}": f"{'u' if sign == 'u' else ''}int{bits}_t"
                for sign in ("i", "u") for bits in (8, 16, 32, 64)}[ty]

    def get_active_torch_device(self):
        raise NotImplementedError("Vortex currently provides compilation only; no Torch execution driver")

    def get_benchmarker(self):
        raise NotImplementedError("Vortex currently provides compilation only; no JIT launcher")
