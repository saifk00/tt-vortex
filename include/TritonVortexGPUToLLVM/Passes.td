include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToVortexLLVM : Pass<"convert-triton-gpu-to-vortex-llvm", "mlir::ModuleOp"> {
  let summary = "Lower supported TritonGPU kernels to Vortex LLVM";
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::arith::ArithDialect"];
}
