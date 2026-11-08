cmake_minimum_required(VERSION 3.26)
project(TritonVortex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
  "import importlib.util; from importlib.metadata import version; print(importlib.util.find_spec('triton').submodule_search_locations[0]); print(version('triton'))"
  OUTPUT_VARIABLE TRITON_INFO OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" TRITON_INFO "${TRITON_INFO}")
list(GET TRITON_INFO 0 TRITON_ROOT)
list(GET TRITON_INFO 1 TRITON_VERSION)
if(NOT TRITON_VERSION STREQUAL "3.9.0" OR NOT EXISTS "${TRITON_ROOT}/include/triton/Tools/PluginUtils.h")
  message(FATAL_ERROR "Use the pinned Triton 3.9.0 development wheel; remove source-checkout PYTHONPATH overrides.")
endif()

set(LLVM_SYSPATH "" CACHE PATH "Extracted pinned LLVM/MLIR SDK")
if(NOT EXISTS "${LLVM_SYSPATH}/bin/mlir-tblgen")
  message(FATAL_ERROR "Set LLVM_SYSPATH to the pinned LLVM SDK, or use make to download it.")
endif()
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

set(PASS_INC "${CMAKE_CURRENT_BINARY_DIR}/include/TritonVortexGPUToLLVM/Passes.h.inc")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/TritonVortexGPUToLLVM")
add_custom_command(OUTPUT "${PASS_INC}"
  COMMAND "${LLVM_SYSPATH}/bin/mlir-tblgen" -gen-pass-decls --name TritonVortexGPUToLLVM
    -I "${LLVM_SYSPATH}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include/TritonVortexGPUToLLVM/Passes.td"
    -o "${PASS_INC}"
  DEPENDS include/TritonVortexGPUToLLVM/Passes.td
  VERBATIM)

add_library(TTVortexLowering OBJECT
  lib/TritonVortexGPUToLLVM/TritonGPUToLLVM.cpp
  lib/TritonVortexGPUToLLVM/TargetInfo.cpp "${PASS_INC}")
set_target_properties(TTVortexLowering PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(TTVortexLowering PUBLIC include "${CMAKE_CURRENT_BINARY_DIR}/include")
target_include_directories(TTVortexLowering SYSTEM PUBLIC "${TRITON_ROOT}/include" "${LLVM_SYSPATH}/include")
target_compile_options(TTVortexLowering PRIVATE -fno-rtti -fno-exceptions)
target_compile_options(TTVortexLowering PUBLIC
  "$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-fstandalone-debug>")

nanobind_add_module(_C STABLE_ABI NB_STATIC NOSTRIP triton_vortex.cc)
target_link_libraries(_C PRIVATE TTVortexLowering)
target_compile_definitions(_C PRIVATE TRITON_VERSION="${TRITON_VERSION}")
# Resolve LLVM/MLIR/Triton references against the wheel, not a second LLVM instance.
target_link_directories(_C PRIVATE "${TRITON_ROOT}/_C")
target_link_libraries(_C PRIVATE triton)
set(TTVORTEX_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/backend" CACHE PATH "Native extension destination")
set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TTVORTEX_OUTPUT_DIRECTORY}"
  BUILD_RPATH "$ORIGIN/../triton/_C;${TRITON_ROOT}/_C"
  INSTALL_RPATH "$ORIGIN/../triton/_C")

option(TTVORTEX_BUILD_OPT "Build the native MLIR debugging driver" OFF)
if(TTVORTEX_BUILD_OPT)
  # libtriton contains Python bindings too. Satisfy their symbols without
  # starting Python; the driver only calls its native MLIR/Triton interfaces.
  find_package(Python 3.12 REQUIRED COMPONENTS Development.Embed)
  add_executable(ttvortex-opt tools/ttvortex-opt.cpp)
  target_compile_options(ttvortex-opt PRIVATE -fno-rtti -fno-exceptions)
  target_link_directories(ttvortex-opt PRIVATE "${TRITON_ROOT}/_C")
  target_link_libraries(ttvortex-opt PRIVATE TTVortexLowering
    "${LLVM_SYSPATH}/lib/libMLIROptLib.a"
    triton Python::Python)
  # Resolve against libtriton first, then add only optimizer-driver facilities
  # absent from the wheel (CLI, bytecode writing, debugging, IRDL and plugins).
  target_link_libraries(ttvortex-opt PRIVATE -Wl,--start-group)
  foreach(component MLIRBytecodeWriter MLIRBytecodeOpInterface MLIRDebug MLIRObservers MLIRPass MLIRIR
      MLIRPluginsLib MLIRSupport MLIRIRDL MLIRRemarkStreamer LLVMRemarks LLVMSupport)
    target_link_libraries(ttvortex-opt PRIVATE "${LLVM_SYSPATH}/lib/lib${component}.a")
  endforeach()
  target_link_libraries(ttvortex-opt PRIVATE -Wl,--end-group)
  set_target_properties(ttvortex-opt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    BUILD_RPATH "${TRITON_ROOT}/_C")
endif()
