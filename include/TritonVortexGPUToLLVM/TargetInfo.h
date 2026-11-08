#ifndef TRITON_VORTEX_GPU_TO_LLVM_TARGET_INFO_H
#define TRITON_VORTEX_GPU_TO_LLVM_TARGET_INFO_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::vortex {
Value readCSR(Location loc, RewriterBase &rewriter, uint32_t csr);

class TargetInfo final : public TargetInfoBase {
public:
  explicit TargetInfo(unsigned warpSize) : warpSize(warpSize) {}
  bool supportMaximumMinimum() const override { return false; }
  bool supportVectorizedAtomics() const override { return false; }
  int getSharedAddressSpace() const override { return 0; }
  int getAddressSpace(Attribute addressSpace) const override;
  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;
  void barrier(Location loc, RewriterBase &rewriter,
               gpu::AddrSpace targets) const override;
  void warpSync(Location loc, RewriterBase &rewriter) const override;
  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Value val, Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;
  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;
  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override;
  bool warpReduce(RewriterBase &, Location, SmallVector<Value> &,
                  triton::ReduceOp, unsigned, unsigned) const override {
    return false;
  }

  Value ballot(RewriterBase &, Location, Type, Value) const override {
    llvm::report_fatal_error("Vortex ballot lowering is not implemented");
  }
  Value getGlobalTimer(RewriterBase &, Location) const override {
    llvm::report_fatal_error("Vortex timer lowering is not implemented");
  }
  StringRef getAtomicSyncScope(MemSyncScope) const override {
    llvm::report_fatal_error("Vortex atomic lowering is not implemented");
  }
  Value loadRelaxed(RewriterBase &, Location, Value, Type, Value,
                    MemSyncScope) const override {
    llvm::report_fatal_error("Vortex atomic lowering is not implemented");
  }
  void storeRelaxed(RewriterBase &, Location, Value, Value, Value,
                    MemSyncScope) const override {
    llvm::report_fatal_error("Vortex atomic lowering is not implemented");
  }
  void clusterBarrier(Location, RewriterBase &, Operation *) const override {
    llvm::report_fatal_error("Vortex cross-CTA lowering is not implemented");
  }
  Value permute(RewriterBase &, Location, Value, Value, Value) const override {
    llvm::report_fatal_error("Vortex permute lowering is not implemented");
  }
  void printf(RewriterBase &, Value, int, ValueRange,
              ArrayRef<bool> = {}) const override {
    llvm::report_fatal_error("Vortex printf lowering is not implemented");
  }
  void printf(RewriterBase &, StringRef, ValueRange,
              ArrayRef<bool> = {}) const override {
    llvm::report_fatal_error("Vortex printf lowering is not implemented");
  }
  void assertFail(RewriterBase &, Location, StringRef, StringRef, StringRef,
                  int) const override {
    llvm::report_fatal_error("Vortex assert lowering is not implemented");
  }

private:
  Value shuffle(RewriterBase &rewriter, Location loc, Value val, Value index,
                unsigned mode) const;
  unsigned warpSize;
};
} // namespace mlir::triton::vortex
#endif
