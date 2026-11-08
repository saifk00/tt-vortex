#include "TritonVortexGPUToLLVM/TargetInfo.h"
#include "TritonVortexGPUToLLVM/CSR.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <string>

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::vortex {
Value readCSR(Location loc, RewriterBase &rewriter, uint32_t csr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return LLVM::InlineAsmOp::create(
             rewriter, loc, rewriter.getI32Type(), ValueRange{b.i32_val(csr)},
             "csrr $0, $1", "=r,i", false, false, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                       LLVM::AsmDialect::AD_ATT),
             ArrayAttr())
      .getResult(0);
}

static Value inlineAsm(RewriterBase &rewriter, Location loc, Type type,
                       ValueRange args, StringRef code, StringRef constraints,
                       bool sideEffects = false) {
  auto op = LLVM::InlineAsmOp::create(
      rewriter, loc, type, args, code, constraints, sideEffects, false,
      LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                LLVM::AsmDialect::AD_ATT),
      ArrayAttr());
  return type ? op.getResult(0) : Value();
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (isa<gpu::SharedMemorySpaceAttr>(addressSpace))
    return getSharedAddressSpace();
  llvm::report_fatal_error("unsupported Vortex address space");
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return TritonLLVMOpBuilder(loc, rewriter).i32_val(0);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc, ModuleOp,
                            ProgramIDDim axis) const {
  return readCSR(loc, rewriter, BlockIdxX + static_cast<unsigned>(axis));
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         gpu::AddrSpace) const {
  auto id = readCSR(loc, rewriter, CTAId);
  auto count = readCSR(loc, rewriter, CTASize);
  inlineAsm(rewriter, loc, Type(), ValueRange{id, count},
            ".insn r 0x0b, 4, 0, x0, $0, $1", "r,r,~{memory}", true);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  barrier(loc, rewriter, gpu::AddrSpace::Local);
}

// Shared-memory predicates select unique writers, so they must guard the
// access.
static Value sharedAccess(RewriterBase &rewriter, Location loc, Value ptr,
                          Type type, Value value, Value pred) {
  auto access = [&]() -> Value {
    if (value) {
      LLVM::StoreOp::create(rewriter, loc, value, ptr);
      return {};
    }
    return LLVM::LoadOp::create(rewriter, loc, type, ptr);
  };
  if (!pred || matchPattern(pred, m_One()))
    return access();
  Block *head = rewriter.getInsertionBlock();
  Block *merge = rewriter.splitBlock(head, rewriter.getInsertionPoint());
  Value result;
  if (!value)
    result = merge->addArgument(type, loc);
  Block *active = rewriter.createBlock(merge);
  Value loaded = access();
  LLVM::BrOp::create(rewriter, loc, loaded ? ValueRange{loaded} : ValueRange{},
                     merge);
  rewriter.setInsertionPointToEnd(head);
  Value other;
  if (!value)
    other = LLVM::UndefOp::create(rewriter, loc, type);
  LLVM::CondBrOp::create(rewriter, loc, pred, active, ValueRange{}, merge,
                         other ? ValueRange{other} : ValueRange{});
  rewriter.setInsertionPointToStart(merge);
  return result;
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Value val, Value pred) const {
  if (ctaId)
    llvm::report_fatal_error("Vortex cross-CTA stores are not implemented");
  sharedAccess(rewriter, loc, ptr, val.getType(), val, pred);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Type elemTy, Value pred,
                              Operation *) const {
  if (ctaId)
    llvm::report_fatal_error("Vortex cross-CTA loads are not implemented");
  return sharedAccess(rewriter, loc, ptr, elemTy, {}, pred);
}

Value TargetInfo::shuffle(RewriterBase &rewriter, Location loc, Value val,
                          Value index, unsigned mode) const {
  if (!val.getType().isInteger(32))
    llvm::report_fatal_error("Vortex shuffles currently require i32");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned clamp = warpSize - 1;
  auto control =
      b.or_(index, b.i32_val((((~clamp) & 0x3f) << 12) | (clamp << 6)));
  // Like vx_shfl_bfly's volatile asm, keep all participating lanes executing.
  return inlineAsm(rewriter, loc, val.getType(), ValueRange{val, control},
                   ".insn r 0x0b, " + std::to_string(mode) + ", 1, $0, $1, $2",
                   "=r,r,r", true);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return shuffle(rewriter, loc, val,
                 TritonLLVMOpBuilder(loc, rewriter).i32_val(i), 6);
}
Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return shuffle(rewriter, loc, val,
                 TritonLLVMOpBuilder(loc, rewriter).i32_val(i), 4);
}
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return shuffleIdx(rewriter, loc, val,
                    TritonLLVMOpBuilder(loc, rewriter).i32_val(i));
}
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return shuffle(rewriter, loc, val, i, 7);
}
} // namespace mlir::triton::vortex
