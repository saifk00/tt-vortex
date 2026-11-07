#ifndef TRITON_VORTEX_GPU_TO_LLVM_CSR_H
#define TRITON_VORTEX_GPU_TO_LLVM_CSR_H

#include <cstdint>

namespace mlir::triton::vortex {
constexpr uint32_t ThreadIdxX = 0xcd3;
constexpr uint32_t BlockIdxX = 0xcd6;
constexpr uint32_t GridDimX = 0xcdc;
} // namespace mlir::triton::vortex

#endif
