"""Triton JIT execution adapter for the Vortex Python runtime."""

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
import os
import struct


def pack_arguments(signature, arguments, device=None):
    """Match add_launch_wrapper: natural scalar alignment, u64 pointers, 8-byte tail."""
    formats = {"i8": "b", "u8": "B", "i16": "h", "u16": "H",
               "i32": "i", "u32": "I", "i64": "q", "u64": "Q"}
    blob = bytearray()
    for index, ty in enumerate(signature.values()):
        if ty == "constexpr":
            continue
        value = arguments[index]
        if ty.startswith("*"):
            from .runtime import Buffer
            if value is None:
                value = 0
            elif isinstance(value, Buffer):
                if device is not None and value.device is not device:
                    raise ValueError("Kernel buffer belongs to another Vortex device")
                value = value.data_ptr()
            elif not isinstance(value, int):
                raise TypeError("Vortex pointers must be device Buffers or integer device addresses")
            fmt = "Q"
        else:
            if ty not in formats:
                raise TypeError(f"Unsupported Vortex scalar argument type: {ty}")
            fmt = formats[ty]
        size = struct.calcsize(fmt)
        blob.extend(b"\0" * (-len(blob) % size))
        blob.extend(struct.pack("<" + fmt, value))
    blob.extend(b"\0" * (-len(blob) % 8))
    return bytes(blob)


class VortexLauncher:
    def __init__(self, src, metadata):
        self.signature = src.signature
        self.metadata = metadata

    def __call__(self, gx, gy, gz, stream, function, packed_metadata, launch_metadata,
                 enter_hook, exit_hook, *args):
        if min(gx, gy, gz) < 0 or max(gx, gy, gz) > 0xffffffff:
            raise ValueError("Grid dimensions must fit unsigned 32-bit integers")
        if not gx or not gy or not gz:
            return
        device = function.device
        if stream != device.queue.value:
            raise ValueError("Kernel and launch queue belong to different devices")
        blob = pack_arguments(self.signature, args, device)
        if len(blob) != self.metadata.args_size:
            raise ValueError(f"Argument ABI mismatch: packed {len(blob)}, expected {self.metadata.args_size}")
        if enter_hook:
            enter_hook(launch_metadata)
        function.launch((gx, gy, gz), self.metadata.num_warps * self.metadata.warp_size,
                        blob, self.metadata.shared)
        if exit_hook:
            exit_hook(launch_metadata)


class VortexUtils:
    def __init__(self, driver):
        self.driver = driver

    def get_device_properties(self, device):
        dev = self.driver._device()
        return {"max_shared_mem": dev.properties["local_mem_size"]}

    def load_binary(self, name, image, shared, device):
        from .runtime import _Module
        dev = self.driver._device()
        module = _Module(dev, image)
        maximum = dev.properties["warp_size"] * dev.properties["num_warps"]
        return module, module, 0, 0, maximum

    def unload_module(self, module):
        if module.handle:
            module.device.synchronize()
        module.close()


class VortexDriver(DriverBase):
    launcher_cls = VortexLauncher

    def __init__(self, device=None):
        self.device = device
        self.utils = VortexUtils(self)

    def _device(self):
        if self.device is None:
            raise RuntimeError("Use `with tt_vortex.runtime.Device() as device:` for Vortex JIT execution")
        self.device._check_open()
        return self.device

    @classmethod
    def is_active(cls):
        # Never replace a CUDA/HIP execution driver merely because Vortex is installed.
        return os.environ.get("TRITON_DEFAULT_BACKEND") == "vortex"

    def get_current_target(self):
        return GPUTarget("vortex", 32, self.device.properties["warp_size"] if self.device else 32)

    def get_current_device(self):
        return self._device()._id

    def get_current_stream(self, device):
        return self._device().queue.value

    def map_python_to_cpp_type(self, ty):
        if ty.startswith("*"):
            return "uint64_t"
        return {f"{sign}{bits}": f"{'u' if sign == 'u' else ''}int{bits}_t"
                for sign in ("i", "u") for bits in (8, 16, 32, 64)}[ty]

    def get_active_torch_device(self):
        raise NotImplementedError("Use Vortex Buffers and NumPy transfers; Torch devices are not supported")

    def get_benchmarker(self):
        raise NotImplementedError("Use Device.snapshot() for simulator cycle measurements")
