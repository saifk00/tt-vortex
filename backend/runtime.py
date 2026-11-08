"""Small synchronous host API for the Vortex runtime (no Torch dependency).

Use Device as a context manager. Transfers finish before returning; kernel
launches are queued until synchronize(), a transfer, or context exit.
The native runtime selects one backend per process.
"""

import ctypes as ct
from collections.abc import Mapping
from functools import lru_cache
from itertools import count
import os
from pathlib import Path
from types import MappingProxyType
import tomllib

import numpy as np
import triton.language as tl

from .compiler import sdk_paths


class _QueueInfo(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("next", ct.c_void_p),
                ("priority", ct.c_int), ("flags", ct.c_uint32)]


class _LaunchInfo(ct.Structure):
    _fields_ = [("struct_size", ct.c_size_t), ("next", ct.c_void_p),
                ("kernel", ct.c_void_p), ("args_host", ct.c_void_p),
                ("args_size", ct.c_size_t), ("ndim", ct.c_uint32),
                ("grid_dim", ct.c_uint32 * 3), ("block_dim", ct.c_uint32 * 3),
                ("lmem_size", ct.c_uint32), ("cluster_dim", ct.c_uint32 * 3)]


_loaded = None
_device_ids = count()


@lru_cache(maxsize=None)
def _counter_definitions(home):
    with (home / "VX_types.toml").open("rb") as source:
        types = tomllib.load(source)
    definitions = {
        "cycles": (types["dcr_mpm_class"]["VX_DCR_MPM_CLASS_BASE"], types["csr_mpm_base"]["VX_CSR_MCYCLE"]),
        "warp_instructions": (types["dcr_mpm_class"]["VX_DCR_MPM_CLASS_BASE"], types["csr_mpm_base"]["VX_CSR_MINSTRET"]),
    }
    for table, counters in types.items():
        if not table.startswith("csr_mpm_") or table == "csr_mpm_base":
            continue
        group = table.removeprefix("csr_mpm_")
        class_name = "VX_DCR_MPM_CLASS_" + group.upper()
        if class_name not in types["dcr_mpm_class"]:
            continue
        for name, address in counters.items():
            if isinstance(address, int) and 0xB00 <= address < 0xB20:
                short_name = name.removeprefix("VX_CSR_MPM_").removeprefix(group.upper() + "_")
                definitions[group + "." + short_name] = (types["dcr_mpm_class"][class_name], address)
    return MappingProxyType(definitions)


class CounterSnapshot(Mapping):
    """Read-only counter values from one device/core; subtract with after.delta(before).

    Counters are cumulative. Deltas cover all device activity between snapshots,
    including any transfers or image reloads in that interval. A counter decrease
    is treated as a reset and rejected rather than reported as a wrapped delta.
    """

    def __init__(self, values, device_id, core):
        self._values = MappingProxyType(dict(values))
        self._device_id, self.core = device_id, core

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def delta(self, before):
        if not isinstance(before, CounterSnapshot):
            raise TypeError("Expected a CounterSnapshot")
        if (self._device_id, self.core) != (before._device_id, before.core):
            raise ValueError("Snapshots must belong to the same device and core")
        if self.keys() != before.keys():
            raise ValueError("Snapshots must contain the same counters")
        result = {name: value - before[name] for name, value in self.items()}
        if any(value < 0 for value in result.values()):
            raise ValueError("A counter decreased between snapshots; the device may have reset")
        return result


def _library(directory, driver):
    global _loaded
    directory = Path(directory).resolve()
    identity = (directory, driver)
    if _loaded is not None:
        if _loaded[0] != identity:
            raise RuntimeError("Vortex selects one runtime/backend per process; use a separate process to switch")
        return _loaded[1]
    if driver not in ("simx", "rtlsim", "firesim"):
        raise ValueError(f"Unsupported Vortex driver: {driver}")
    # Preload by absolute path: the stub subsequently dlopens this same basename.
    backend = ct.CDLL(str(directory / f"libvortex-{driver}.so"), mode=ct.RTLD_GLOBAL)
    lib = ct.CDLL(str(directory / "libvortex.so"))
    p, u32, u64 = ct.c_void_p, ct.c_uint32, ct.c_uint64
    signatures = {
        "vx_result_string": ([ct.c_int], ct.c_char_p),
        "vx_device_open": ([u32, ct.POINTER(p)], ct.c_int),
        "vx_device_release": ([p], ct.c_int),
        "vx_device_query": ([p, u32, ct.POINTER(u64)], ct.c_int),
        "vx_device_mpm_query": ([p, u32, u32, u32, ct.POINTER(u64)], ct.c_int),
        "vx_queue_create": ([p, ct.POINTER(_QueueInfo), ct.POINTER(p)], ct.c_int),
        "vx_queue_release": ([p], ct.c_int),
        "vx_queue_finish": ([p, u64], ct.c_int),
        "vx_buffer_create": ([p, u64, u32, ct.POINTER(p)], ct.c_int),
        "vx_buffer_release": ([p], ct.c_int),
        "vx_buffer_address": ([p, ct.POINTER(u64)], ct.c_int),
        "vx_enqueue_write": ([p, p, u64, p, u64, u32, p, p], ct.c_int),
        "vx_enqueue_read": ([p, p, p, u64, u64, u32, p, p], ct.c_int),
        "vx_module_load_bytes": ([p, p, ct.c_size_t, ct.POINTER(p)], ct.c_int),
        "vx_module_get_kernel": ([p, ct.c_char_p, ct.POINTER(p)], ct.c_int),
        "vx_module_release": ([p], ct.c_int),
        "vx_kernel_release": ([p], ct.c_int),
        "vx_enqueue_launch": ([p, ct.POINTER(_LaunchInfo), u32, p, p], ct.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    os.environ["VORTEX_DRIVER"] = driver
    _loaded = (identity, lib, backend)
    return lib


class Device:
    """Own a runtime device and queue; inside `with`, select it for Triton JIT.

    `runtime_dir` defaults to $VORTEX_BUILD_DIR/sw/runtime. Simulator and compiler
    must use matching SDK configurations. Device contexts may be nested, but
    this initial API is intended for a single Python host thread.
    """

    def __init__(self, driver=None, index=0, runtime_dir=None):
        self.lib = _library(runtime_dir or sdk_paths()[3] / "sw/runtime",
                            driver or os.environ.get("VORTEX_DRIVER", "simx"))
        self.handle, self.queue = ct.c_void_p(), ct.c_void_p()
        self._buffers, self._modules = [], []
        self._resident_module = None
        self._id = next(_device_ids)
        self._entered = False
        self._call("vx_device_open", index, ct.byref(self.handle))
        try:
            info = _QueueInfo(struct_size=ct.sizeof(_QueueInfo))
            self._call("vx_queue_create", self.handle, ct.byref(info), ct.byref(self.queue))
            self.properties = {name: self.query(cap) for name, cap in
                               (("warp_size", 1), ("num_warps", 2), ("num_cores", 3),
                                ("cache_line_size", 4), ("local_mem_size", 6))}
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            message = self.lib.vx_result_string(result).decode()
            raise RuntimeError(f"{name}: {message} ({result})")

    def _check_open(self):
        if not self.handle:
            raise RuntimeError("Vortex device is closed")

    def query(self, capability):
        self._check_open()
        value = ct.c_uint64()
        self._call("vx_device_query", self.handle, capability, ct.byref(value))
        return value.value

    def synchronize(self):
        self._check_open()
        self._call("vx_queue_finish", self.queue, ct.c_uint64(-1))

    def empty(self, shape, dtype=np.int32):
        self._check_open()
        return Buffer(self, shape, dtype)

    def from_numpy(self, array):
        array = np.ascontiguousarray(array)
        buffer = self.empty(array.shape, array.dtype)
        buffer.copy_from(array)
        return buffer

    def full(self, shape, value, dtype=np.int32):
        return self.from_numpy(np.full(shape, value, dtype=dtype))

    @property
    def counter_definitions(self):
        """SDK counter names mapped to (MPM class, CSR address); not a capability list."""
        return _counter_definitions(sdk_paths()[0])

    def counter(self, address, group=0, core=0):
        """Read a named or raw MPM counter without synchronizing.

        Names, e.g. ``core.DIVERGENCE``, select the class automatically. Raw
        addresses use ``group``. ``core`` selects one core; shared cache counters
        describe its socket/cluster and must not be summed again for every core.
        The raw-address API also accepts native core ID 0xffffffff to sum cores.
        Extended counters require a PERF-enabled runtime. Disabled units may
        return zero; the C API does not distinguish that from no events.
        """
        self._check_open()
        aggregate = not isinstance(address, str) and core == 0xffffffff
        if not isinstance(core, int) or not (0 <= core < self.properties["num_cores"] or aggregate):
            raise ValueError("core must select an existing core")
        if isinstance(address, str):
            if group != 0:
                raise ValueError("Named counters select their own group")
            group, address = self.counter_definitions[address]
        value = ct.c_uint64()
        self._call("vx_device_mpm_query", self.handle, group, address, core, ct.byref(value))
        return value.value

    def snapshot(self, counters=None, *, core=0):
        """Finish queued work and read named counters for one core (default 0).

        The default includes cycles/instructions, core execution and stall
        counters, and L1 instruction/data caches. Pass an iterable of names to
        select a subset or other classes from ``counter_definitions``. Reading
        uses the host MPM port; VORTEX_PROFILING does not select these counters.
        """
        if not isinstance(core, int) or not 0 <= core < self.properties["num_cores"]:
            raise ValueError("core must select an existing core")
        if counters is None:
            counters = [name for name in self.counter_definitions
                        if "." not in name or name.startswith(("core.", "icache.", "dcache."))]
        elif isinstance(counters, str):
            counters = [counters]
        else:
            counters = list(counters)
        for name in counters:
            if name not in self.counter_definitions:
                raise KeyError(f"Unknown performance counter: {name}")
        self.synchronize()
        return CounterSnapshot({name: self.counter(name, core=core) for name in counters}, self._id, core)

    def __enter__(self):
        from triton.runtime import driver
        from .driver import VortexDriver
        self._check_open()
        if self._entered:
            raise RuntimeError("Device context is already active")
        self._previous_driver = driver._active
        self._triton_driver = VortexDriver(self)
        driver.set_active(self._triton_driver)
        self._entered = True
        return self

    def close(self):
        if self.handle:
            try:
                if self.queue:
                    self.synchronize()
            finally:
                try:
                    for module in reversed(self._modules):
                        module.close()
                    for buffer in reversed(self._buffers):
                        buffer.close(synchronize=False)
                    if self.queue:
                        self._call("vx_queue_release", self.queue)
                        self.queue = ct.c_void_p()
                    self._call("vx_device_release", self.handle)
                    self.handle = ct.c_void_p()
                finally:
                    if self._entered:
                        from triton.runtime import driver
                        driver._active = self._previous_driver
                        self._entered = False

    def __exit__(self, *exc):
        self.close()


class Buffer:
    """Typed device allocation; data_ptr() and dtype implement Triton's protocol."""

    def __init__(self, device, shape, dtype):
        self.device = device
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if any(not isinstance(n, int) or n < 0 for n in self.shape):
            raise ValueError("Buffer dimensions must be nonnegative integers")
        self.numpy_dtype = np.dtype(dtype)
        if not self.numpy_dtype.isnative:
            raise ValueError("Only native-endian buffer dtypes are supported")
        types = {np.dtype(name): getattr(tl, name) for name in
                 ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                  "float16", "float32", "float64")}
        if self.numpy_dtype not in types:
            raise TypeError(f"Unsupported buffer dtype: {self.numpy_dtype}")
        self.dtype = types[self.numpy_dtype]
        self.nbytes = int(np.prod(self.shape)) * self.numpy_dtype.itemsize
        self.handle = ct.c_void_p()
        device._call("vx_buffer_create", device.handle, max(1, self.nbytes), 3, ct.byref(self.handle))
        device._buffers.append(self)
        address = ct.c_uint64()
        device._call("vx_buffer_address", self.handle, ct.byref(address))
        self._address = address.value

    def data_ptr(self):
        self.device._check_open()
        if not self.handle:
            raise RuntimeError("Vortex buffer is closed")
        return self._address

    def copy_from(self, array):
        self.data_ptr()
        array = np.ascontiguousarray(array)
        if array.shape != self.shape or array.dtype != self.numpy_dtype:
            raise ValueError("Transfer shape and dtype must match the device buffer")
        if self.nbytes:
            self.device._call("vx_enqueue_write", self.device.queue, self.handle, 0,
                              array.ctypes.data, self.nbytes, 0, None, None)
        # Keep the host array alive until the queued write has finished.
        self.device.synchronize()

    def numpy(self):
        self.data_ptr()
        array = np.empty(self.shape, dtype=self.numpy_dtype)
        if self.nbytes:
            self.device._call("vx_enqueue_read", self.device.queue, array.ctypes.data,
                              self.handle, 0, self.nbytes, 0, None, None)
        self.device.synchronize()
        return array

    def fill(self, value):
        self.copy_from(np.full(self.shape, value, dtype=self.numpy_dtype))

    def close(self, synchronize=True):
        if self.handle:
            if synchronize:
                self.device.synchronize()
            self.device._call("vx_buffer_release", self.handle)
            self.handle = ct.c_void_p()


class _Module:
    def __init__(self, device, image):
        self.device = device
        self._image, self._closed = image, False
        self.handle, self.kernel = ct.c_void_p(), ct.c_void_p()
        device._modules.append(self)
        self._ensure_loaded()

    def _ensure_loaded(self):
        self.device._check_open()
        if self._closed:
            raise RuntimeError("Vortex kernel is unloaded")
        if self.handle:
            return
        # VXBINs currently link at the same fixed startup address. Keep compiled
        # images cached on the host, but only one resident on the device.
        self.device.synchronize()
        resident = self.device._resident_module
        if resident is not None:
            resident._unload()
        data = ct.create_string_buffer(self._image)
        self.device._call("vx_module_load_bytes", self.device.handle, data,
                          len(self._image), ct.byref(self.handle))
        try:
            self.device._call("vx_module_get_kernel", self.handle, b"main", ct.byref(self.kernel))
        except BaseException:
            self._unload()
            raise
        self.device._resident_module = self

    def _unload(self):
        if self.kernel:
            self.device._call("vx_kernel_release", self.kernel)
            self.kernel = ct.c_void_p()
        if self.handle:
            self.device._call("vx_module_release", self.handle)
            self.handle = ct.c_void_p()
        if self.device._resident_module is self:
            self.device._resident_module = None

    def close(self):
        self._unload()
        self._closed = True

    def launch(self, grid, block, blob, shared=0):
        self._ensure_loaded()
        data = ct.create_string_buffer(blob)
        info = _LaunchInfo(struct_size=ct.sizeof(_LaunchInfo), kernel=self.kernel,
                           args_host=ct.cast(data, ct.c_void_p), args_size=len(blob), ndim=3,
                           grid_dim=(ct.c_uint32 * 3)(*grid),
                           block_dim=(ct.c_uint32 * 3)(block, 1, 1), lmem_size=shared)
        # vx_enqueue_launch copies the argument blob before returning.
        self.device._call("vx_enqueue_launch", self.device.queue, ct.byref(info), 0, None, None)
