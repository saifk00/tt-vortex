.DEFAULT_GOAL := native
PYTHON ?= python3
CXX = clang++
JOBS ?= 4
BUILD_TYPE ?= Debug
BUILD_OPT ?= OFF
LLVM_SYSPATH ?= $(CURDIR)/.cache/llvm
IR_TESTS ?= tests/ir
LIT_ARGS ?= -v

.PHONY: native opt configure sdk install install-templates wheel test check-ir clean
sdk:
	$(PYTHON) ci/fetch_llvm.py $(LLVM_SYSPATH)

configure: sdk
	$(PYTHON) -m cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DCMAKE_CXX_COMPILER=$(CXX) -DPython_EXECUTABLE=$(shell $(PYTHON) -c 'import sys; print(sys.executable)') \
	    -DLLVM_SYSPATH=$(LLVM_SYSPATH) -DTTVORTEX_BUILD_OPT=$(BUILD_OPT)

native: configure
	$(PYTHON) -m cmake --build build --parallel $(JOBS)

opt: BUILD_OPT=ON
opt: configure
	$(PYTHON) -m cmake --build build --target ttvortex-opt --parallel $(JOBS)

check-ir: opt
	LLVM_SYSPATH=$(LLVM_SYSPATH) TTVORTEX_OPT=$(CURDIR)/build/bin/ttvortex-opt \
	    $(PYTHON) -c 'from lit.main import main; main()' $(LIT_ARGS) $(IR_TESTS)

install: sdk
	LLVM_SYSPATH=$(LLVM_SYSPATH) CXX=$(CXX) CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) \
	    uv pip install --python $(PYTHON) --no-build-isolation -e .
	$(MAKE) install-templates

# Build the source archive, then compile the wheel from that archive.
wheel: sdk
	LLVM_SYSPATH=$(LLVM_SYSPATH) CXX=$(CXX) CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) $(PYTHON) -m build

# Triton's AOT CLI opens templates by filesystem path, bypassing entry points.
# Editable Python installs therefore need these files in the installed Triton tree.
install-templates:
	$(PYTHON) -c 'import importlib.util,pathlib,shutil; dst=pathlib.Path(importlib.util.find_spec("triton").origin).parent/"tools/extra/vortex"; dst.mkdir(parents=True,exist_ok=True); [shutil.copy2(src,dst/src.name) for src in pathlib.Path("tools/vortex").glob("compile.*")]'

test: native install-templates
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf build backend/_C*.so
