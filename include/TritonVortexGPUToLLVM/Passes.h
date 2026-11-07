#ifndef TRITON_VORTEX_GPU_TO_LLVM_PASSES_H
#define TRITON_VORTEX_GPU_TO_LLVM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::vortex {
#define GEN_PASS_DECL
#include "vortex/include/TritonVortexGPUToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "vortex/include/TritonVortexGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::vortex

#endif
