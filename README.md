# tt-vortex: experimental Triton backend

[![CI](https://github.com/saifk00/tt-vortex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saifk00/tt-vortex/actions/workflows/ci.yml)

An external Triton backend for the Vortex RISC-V GPU. It supplies TTIR/TTGIR
lowering to RV32 LLVM, VXBIN packaging, and AOT host launcher templates.
Simple integer copy/add kernels compile and run through the Vortex runtime on
SimX and processor RTL simulation. This is an experimental baseline, not a
complete Triton implementation.

## Tested dependencies

- Published Triton 3.9.0 wheel (source revision [`b7ddcf86`](https://github.com/triton-lang/triton/commit/b7ddcf86c095de79527ba3098503e73158bb94bd)), with native extension support and headers.
- Vortex [`50cb1745`](https://github.com/vortexgpgpu/vortex/commit/50cb174546ec69a3869b42b8871e932e4440511f), runtime API in `vortex2.h` and a configured RV32 build.
- Vortex toolchain `v3.0.1`, including Vortex Clang 20.1.8 (`a81a5957`).
- Linux x86-64 host, Python 3.13, and GCC/Clang supporting C++17 and GNU assembly labels.

Triton native APIs evolve quickly; these revisions record the tested combination,
not a compatibility guarantee for other releases. The simulated add sweep used
one core, four 32-thread warps, and a 16 KiB L1 D-cache.

## Continuous integration

[GitHub Actions](.github/workflows/ci.yml) runs on pushes to `main`, pull requests,
and manual dispatch. It uses an Ubuntu 24.04 x86-64 hosted runner with Python 3.13;
no GPU, self-hosted runner, or repository secrets are needed. The job:

1. Installs the pinned published Triton wheel and builds our standalone extension.
2. Runs the compiler/launcher tests through Triton's normal API and AOT CLI.
3. AOT-compiles [add](examples/add.py), builds its [host benchmark](examples/add.cpp),
   and runs three SimX launches each for N=1024 and N=1001 (masked tail).
4. Checks every output element and the untouched guard region, and requires
   nonzero cycle/instruction counters. Logs, generated launchers, compiler IR,
   and performance CSVs are uploaded as `simx-results` for 14 days.

Cycles and stall/divergence counters are recorded for inspection, not enforced
against a performance threshold. This checks functional codegen on SimX;
processor RTL simulation remains a separate local check.

The first run downloads both toolchains and builds SimX plus its dependencies.
Later runs restore the LLVM SDK and matching Vortex runtime from Actions caches.
Cache keys include dependency pins and build recipes; cache misses rebuild from
the pinned sources. The plugin itself is built fresh on every run.

To reproduce with an activated Python environment and host Make/GCC/Clang:

```bash
python -m pip install -r ci/requirements.txt 'triton==3.9.0'
python ci/prepare_vortex.py --jobs 2
source .cache/vortex-sdk/env.sh
make native
python -m pip install --no-deps --no-build-isolation -e .
make install-templates
bash ci/test.sh
```

An existing compatible SDK can instead supply `VORTEX_HOME`, `VORTEX_TOOLDIR`,
`VORTEX_BUILD_DIR`, and `CONFIGS`, skipping `prepare_vortex.py`. The benchmark
expects one core, four hardware warps, and 32 threads per warp. The cached SDK
contains absolute build paths; delete `.cache/vortex-sdk` before relocating it
or changing dependency pins locally.

Enable Actions in the repository settings if disabled. The workflow needs only
`contents: read`; its action dependencies are pinned to commit hashes. Once the
first run passes, the badge above turns green. Branch protection may optionally
require the `Compile and simulate` check.

## Dependency inputs

[ci/dependencies.lock.json](ci/dependencies.lock.json) records the selected source
commits, Triton's wheel and LLVM/MLIR URLs and SHA-256 checksums, Vortex's prebuilt
toolchain commit, Python version, simulator configuration, and Ramulator's
transitive source revisions. The SDK preparation scripts consume this manifest;
the CI wheel download also verifies its SHA-256.

Both LLVM toolchains can be downloaded rather than built. They are distinct:
Triton's LLVM/MLIR builds the compiler plugin, while Vortex LLVM generates the
RV32 device image. Vortex's `v3.0.1` toolchain tag is movable, so downloads must
use the recorded commit, not resolve the label anew on every run.

The native extension is built independently against the published Triton wheel.
The wheel exports Triton/LLVM/MLIR symbols and supplies generated Triton headers;
the LLVM SDK supplies the remaining headers and TableGen. We do not rebuild or
relink Triton or LLVM. Python discovers the backend through a `triton.backends`
entry point, and the backend loads the native lowering with
`passes.plugin.extend_with()`. The same shared library exposes the native LLVM
launch-wrapper helpers through nanobind. Its bytes are included in the compiler
cache key, so rebuilding the lowering invalidates cached VXBINs.

Upstream Vortex CI uploads build artifacts, but its default four-thread warp
configuration does not match this project's 32-thread warp. Those artifacts are
not a drop-in simulator bundle for this example. Our job builds and caches its
own matching runtime, generated headers, source files, and dependencies. No
release artifact publishing or deployment is configured.

## Installation

Use an activated Python 3.12+ environment with a host C++ compiler and Make:

```bash
uv pip install 'triton==3.9.0' 'setuptools>=64' wheel 'cmake>=3.26,<4' ninja nanobind==2.10.2
make install
```

`make install` downloads the pinned LLVM development SDK if needed, builds only
our extension, registers the editable Python package, and installs the AOT
templates. Native output is `backend/_C.abi3.so`; build files and clangd commands
are in `build/`. An existing SDK can be selected with
`make install LLVM_SYSPATH=/path/to/pinned/llvm`.

For subsequent native edits run `make`; Python edits are visible immediately in
the editable install. After template edits run `make install-templates`. AOT
looks up template files by path inside the installed Triton package, so this
small copy step is required for editable installs. Normal wheels include those
files directly. To package after the native build, run
`uv build --wheel --no-build-isolation`.

In the parent workspace, `make setup` installs the locked environment and builds
the extension; `make compiler` rebuilds it; `make run DRIVER=simx` compiles and
simulates add. No source-checkout `PYTHONPATH`, `.pth`, or `TRITON_PLUGIN_DIRS`
setting is required. Remove old overrides before invoking Python directly.

Install/configure the Vortex SDK separately using
[Vortex's build instructions](https://github.com/vortexgpgpu/vortex/blob/50cb174546ec69a3869b42b8871e932e4440511f/docs/install_vortex.md).

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
The included host benchmark and `ci/test.sh` demonstrate the complete sequence.

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

The package installs `tools/vortex` as `triton.tools.extra.vortex`, where the
standard AOT CLI looks for its templates. No Triton source edits are needed.

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
make test
```

The 13 tests check discovery, standalone loading, native cache invalidation,
kernel compilation and the host launcher ABI without a GPU or simulator. They
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
