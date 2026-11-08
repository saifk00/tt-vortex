.DEFAULT_GOAL := native
PYTHON ?= python3
CXX = clang++
JOBS ?= 4
BUILD_TYPE ?= Debug
LLVM_SYSPATH ?= $(CURDIR)/.cache/llvm

.PHONY: native configure sdk install install-templates test clean
sdk:
	$(PYTHON) ci/fetch_llvm.py $(LLVM_SYSPATH)

configure: sdk
	$(PYTHON) -m cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DCMAKE_CXX_COMPILER=$(CXX) -DPython_EXECUTABLE=$(shell $(PYTHON) -c 'import sys; print(sys.executable)') \
	    -DLLVM_SYSPATH=$(LLVM_SYSPATH)

native: configure
	$(PYTHON) -m cmake --build build --parallel $(JOBS)

install: native
	uv pip install --python $(PYTHON) --no-build-isolation -e .
	$(MAKE) install-templates

# Triton's AOT CLI opens templates by filesystem path, bypassing entry points.
# Editable Python installs therefore need these files in the installed Triton tree.
install-templates:
	$(PYTHON) -c 'import importlib.util,pathlib,shutil; dst=pathlib.Path(importlib.util.find_spec("triton").origin).parent/"tools/extra/vortex"; dst.mkdir(parents=True,exist_ok=True); [shutil.copy2(src,dst/src.name) for src in pathlib.Path("tools/vortex").glob("compile.*")]'

test: native install-templates
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf build backend/_C*.so
