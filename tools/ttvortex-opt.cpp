// Native driver for the same lowering implementation used by the Python backend.
#include "TritonVortexGPUToLLVM/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect,
                  mlir::triton::gpu::TritonGPUDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::ub::UBDialect>();
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::triton::registerTritonPasses();
  mlir::triton::registerConvertTritonToTritonGPU();
  mlir::registerPass([] { return mlir::createSCFToControlFlowPass(); });
  mlir::registerPass([] { return mlir::createReconcileUnrealizedCastsPass(); });
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  mlir::registerInlinerPass();
  // The shared lowering object registers convert-triton-gpu-to-vortex-llvm.
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton Vortex MLIR optimizer\n", registry));
}
