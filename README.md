# tt-vortex: experimental Triton backend

An external Triton backend for the Vortex RISC-V GPU. It supplies TTIR/TTGIR
lowering to RV32 LLVM, VXBIN packaging, and AOT host launcher templates.
Simple integer copy/add kernels compile and run through the Vortex runtime on
SimX and processor RTL simulation. This is an experimental baseline, not a
complete Triton implementation.

## Tested dependencies

- Triton [`b7ddcf86`](https://github.com/triton-lang/triton/commit/b7ddcf86c095de79527ba3098503e73158bb94bd), built from source with this plugin.
- Vortex [`50cb1745`](https://github.com/vortexgpgpu/vortex/commit/50cb174546ec69a3869b42b8871e932e4440511f), runtime API in `vortex2.h` and a configured RV32 build.
- Vortex toolchain `v3.0.1`, including Vortex Clang 20.1.8 (`a81a5957`).
- Linux x86-64 host, Python 3.13, and GCC/Clang supporting C++17 and GNU assembly labels.

Triton native APIs evolve quickly; these revisions record the tested combination,
not a compatibility guarantee for other releases. The simulated add sweep used
one core, four 32-thread warps, and a 16 KiB L1 D-cache.

## Installation

Triton's supported `TRITON_PLUGIN_DIRS` mechanism compiles the native pass into
`libtriton.so` and installs links for the backend and templates:

```bash
# In an activated virtual environment; install Triton's build requirements first.
uv pip install 'setuptools>=40.8.0' wheel 'cmake>=3.20,<4' 'ninja>=1.11.1' nanobind==2.10.2
TRITON_PLUGIN_DIRS=/absolute/path/to/tt-vortex \
  uv pip install --no-build-isolation -e /absolute/path/to/triton
```

Rebuild the configured Triton CMake tree after native C++ changes. Python backend
and template changes do not need a native rebuild. This is not a pure-Python
plugin for arbitrary prebuilt Triton wheels. Install/configure the Vortex SDK
separately using [Vortex's build instructions](https://github.com/vortexgpgpu/vortex/blob/50cb174546ec69a3869b42b8871e932e4440511f/docs/install_vortex.md).

## Compilation

From this repository, use Triton's standard AOT CLI with the included
[add kernel](examples/add.py):

```bash
export VORTEX_HOME=/absolute/path/to/vortex
export VORTEX_TOOLDIR="$VORTEX_HOME/tools"
export VORTEX_BUILD_DIR="$VORTEX_HOME/build32"  # configured RV32 build
export TRITON_DEFAULT_BACKEND=vortex
export TRITON_STORE_BINARY_ONLY=0
mkdir -p build
python -m triton.tools.compile examples/add.py \
  --kernel-name add_kernel --signature '*i32,*i32,*i32,i32,256' \
  --grid '(n+255)/256,1,1' --num-warps 4 \
  --target vortex:32:32 --out-name add --out-path build/add
c++ -std=c++17 -DTTVORTEX_AOT_SHORT_NAMES -I"$VORTEX_HOME/sw/runtime/include" \
  -c build/add.*.cpp -o build/launcher.o
```

The outputs are `build/add.<specialization>.cpp` and `.h`; the implementation
embeds the VXBIN. Include the generated header in a host program, link
`build/launcher.o` with Vortex's `libvortex`, allocate/upload buffers, and call
the launcher with device addresses and `n`. Finish the queue before reading
results. Select the built runtime driver with `VORTEX_DRIVER=simx` or `rtlsim`.
This repository supplies the compiler and launcher, not a simulator build or
host benchmark harness.

Set `TRITON_DEFAULT_BACKEND=vortex` for AOT compilation, including when using
`--target vortex:32:32`. The CLI takes host type names from the active driver;
the Vortex driver emits `uint64_t` device addresses. This is a compilation-only
driver; Python JIT execution and Torch device integration are not implemented.

The ordinary `triton.compile(ASTSource(...), target=GPUTarget("vortex",32,32))`
API exposes TTIR, TTGIR, LLVM text in `asm["llir"]`, and image bytes in
`asm["vxbin"]`. The stages are:

```text
AST → built-in TTIR cleanup → TTGIR → native Vortex LLVM conversion
    → native launch wrapper → Vortex Clang → startup/link → vxbin.py
```

## Files and ABI

| File | Role |
| --- | --- |
| [backend/compiler.py](backend/compiler.py) | Selection, stages, SDK codegen and image packaging |
| [backend/driver.py](backend/driver.py) | Explicit AOT driver selection and C type names |
| [triton_vortex.cc](triton_vortex.cc) | Pass bindings and LLVM argument wrapper |
| [tools/vortex/compile.cpp](tools/vortex/compile.cpp) | Embedded VXBIN, load/unload and asynchronous queue launch |
| [tools/vortex/compile.h](tools/vortex/compile.h) | Generated C API |
| [TritonGPUToLLVM.cpp](lib/TritonVortexGPUToLLVM/TritonGPUToLLVM.cpp) | Native lowering patterns |
| [CSR.h](include/TritonVortexGPUToLLVM/CSR.h) | CSR definitions |

Triton's installer links `tools/vortex` into `triton.tools.extra.vortex`, just
as NVIDIA's `tools/cuda` becomes `extra/cuda`. No Triton source edits are needed.

The generated C-linkage function takes a Vortex device, queue, and kernel arguments.
Pointer arguments are uint64 device addresses from `vx_buffer_address`.
It lazily loads its embedded image, gets the `main` entry, packs arguments and
enqueues the grid. Call `vx_queue_finish` before reading results or unloading.
Each generated specialization holds one global module/kernel; serialize calls
and use one device until unloaded.

`kernel_main(ptr args)` reads naturally aligned integer slots of 1/2/4/8 bytes
and 8-byte device-address slots. The blob rounds to 8 bytes; RV32 uses the low
32 address bits. Add's ABI is `a@0:u64, b@8:u64, dst@16:u64, n@24:i32`, 32 bytes.
The CLI's unused scratch placeholders are excluded. A C++17 parameter pack
preserves argument types while building this blob; there is no fixed argument
count limit. Compile generated `.cpp` files with a C++17 compiler and link with
the C++ driver. The generated `.h` remains usable by C callers. Current lowerings
support simple integer kernels, without floating-point kernels or shared memory.

Program ID reads CSR `0xcd6 + axis`; thread X reads `0xcd3`. Masked memory uses
LLVM control flow. Generic Triton elementwise, layout and packing helpers are
reused by the native pass.

## SDK and checks

SDK paths default to the sibling Vortex checkout and its tools/build32 tree.
Override with `VORTEX_HOME`, `VORTEX_TOOLDIR`, `VORTEX_LLVM_PATH`, and
`VORTEX_BUILD_DIR` (configured RV32 headers). Leave `TRITON_STORE_BINARY_ONLY`
unset or zero; upstream's binary-only cache hardcodes CUDA/HIP suffixes.
Triton's separate multi-variant AOT linker does not have Vortex templates yet.

With the plugin installed and the SDK environment above, run from this repository:

```bash
python -m unittest discover -s tests -v
```

The tests compile kernels and host launchers without a GPU or simulator. They
use kernels defined in the test file and respect `VORTEX_HOME`, so no parent
workspace examples are required. End-to-end simulation was validated separately
with a host benchmark: BLOCK=128/256/512/1024 × WARPS=1/2/4, N=1024, three launches
per configuration on both SimX and RTL (72 checked launches). A masked tail with
N=1001, BLOCK=256, WARPS=2 also passed on RTL.

## Current limits

- RV32, rank-one blocked integer tensors; no floating-point kernels, atomics,
  reductions, shared memory, general layout conversions or Python JIT execution.
- `BLOCK` is elements per program, not per thread. For a fully populated blocked
  layout, elements per thread are `BLOCK / (32 * num_warps)`. The grid is
  `ceil(n / BLOCK)` and `num_warps` cannot exceed the core's physical warp capacity.
- Redundant stores are not suppressed yet. Keep each logical tensor large enough
  to populate its layout's threads; for the example use `BLOCK >= 32 * num_warps`.
  Underfilled layouts and scalar stores are outside the validated baseline.
- The host ABI assumes a little-endian host. Device pointers occupy 64-bit slots,
  but RV32 consumes only their low 32 bits.
- Launcher state is per specialization, single-device and not thread-safe.
  Finish queued work before unloading. Large blocks can generate heavy stack
  spills; performance tuning is intentionally minimal.

## Comparison with NVIDIA's backend

Both backends implement Triton's `BaseBackend` stages and use the same AOT CLI
to format launcher templates. That scaffolding belongs to Triton's interface;
we don't need to copy NVIDIA's optimization pipeline or its execution driver.

| Area | NVIDIA | Vortex |
| --- | --- | --- |
| Exported AOT names | CLI adds a specialization hash and hint suffix | Same CLI naming; independent of `VortexBackend.hash()` |
| Module loading | Embedded CUBIN, CUDA context/stream, `cuModuleGetFunction` | Embedded VXBIN, explicit device/queue, runtime entry `main` |
| Arguments | Array of addresses; CUDA obtains layout from the compiled image | Host packs a byte blob matching our native `kernel_main` wrapper |
| LLVM lowering | NVVM intrinsics and PTX-specific instructions | RISC-V CSRs (program X: `0xcd6`), generic LLVM operations and masked control flow |
| Final compilation | LLVM → PTX → `ptxas` → CUBIN | Vortex Clang → RV32 object + startup/linker script → ELF → VXBIN |
| Execution integration | Python JIT launch, device discovery, Torch | Compilation-only Python driver; execution through the generated C API |

The AOT specialization hash is useful: two variants must have different linker
symbols. NVIDIA's separate AOT linker generates dispatchers over those symbols;
Vortex does not implement that linker yet. For a single specialization, define
`TTVORTEX_AOT_SHORT_NAMES` before including the generated header. It declares real
functions using the source kernel name (`add_kernel`, `load_add_kernel`,
`unload_add_kernel`) and GNU/Clang `__asm__` labels bind them to the hashed linker
symbols. Also compile the generated implementation with
`-DTTVORTEX_AOT_SHORT_NAMES` so source navigation finds the matching definitions.
This adds no wrappers or runtime overhead, and keeps linker symbols unique.
Without the opt-in, the source functions use hashed names as before. Opting into
short names for two variants of the same source kernel in one translation unit
is a preprocessing error. No generated `entry.h` or symbol scraping is needed.

`VortexBackend.hash()` instead fingerprints codegen and SDK inputs for Triton's
compilation cache. Changing it does not remove the AOT symbol suffix. Unlike
NVIDIA's toolchain-version fingerprint, Vortex also needs to account for its
startup code, linker script, generated configuration headers, and image packager.

Two upstream AOT interface constraints remain explicit: template parameters
provide argument addresses but not their sizes/offsets (hence the typed C++
packer), and C type names come from the active driver even with an explicit
`--target`. Set `TRITON_DEFAULT_BACKEND=vortex` to get `uint64_t` device addresses.
The templates use only Vortex and standard host types. Neither issue needs
a local Triton fork. See upstream [compile.py](https://github.com/triton-lang/triton/blob/b7ddcf86c095de79527ba3098503e73158bb94bd/python/triton/tools/compile.py)
and [CUDA's launcher](https://github.com/triton-lang/triton/blob/b7ddcf86c095de79527ba3098503e73158bb94bd/third_party/nvidia/tools/cuda/compile.c).
