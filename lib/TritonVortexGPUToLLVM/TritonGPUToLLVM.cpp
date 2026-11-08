#include "TritonVortexGPUToLLVM/CSR.h"
#include "TritonVortexGPUToLLVM/Passes.h"
#include "TritonVortexGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton::vortex {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOVORTEXLLVM
#include "TritonVortexGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::vortex

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

class VortexTypeConverter : public LLVMTypeConverter {
public:
  explicit VortexTypeConverter(MLIRContext *ctx)
      : LLVMTypeConverter(ctx, options(ctx)) {
    addConversion([ctx](triton::PointerType) -> Type {
      return LLVM::LLVMPointerType::get(ctx, 0);
    });
    addConversion([this, ctx](RankedTensorType type) -> Type {
      Type element = convertType(type.getElementType());
      return LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(ttg::getUniqueElemsPerThread(type), element));
    });
  }

private:
  static LowerToLLVMOptions options(MLIRContext *ctx) {
    LowerToLLVMOptions options(ctx);
    options.overrideIndexBitwidth(32);
    return options;
  }
};

using triton::vortex::readCSR;

struct GetProgramIdOpConversion
    : ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto csr = triton::vortex::BlockIdxX + static_cast<uint32_t>(op.getAxis());
    rewriter.replaceOp(op, readCSR(op.getLoc(), rewriter, csr));
    return success();
  }
};

struct GetNumProgramsOpConversion
    : ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto csr = triton::vortex::GridDimX + static_cast<uint32_t>(op.getAxis());
    rewriter.replaceOp(op, readCSR(op.getLoc(), rewriter, csr));
    return success();
  }
};

struct ThreadIdOpConversion : ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDimension() != mlir::gpu::Dimension::x)
      return rewriter.notifyMatchFailure(op, "only a 1D block is supported");
    rewriter.replaceOp(
        op, readCSR(op.getLoc(), rewriter, triton::vortex::ThreadIdxX));
    return success();
  }
};

struct WarpIdOpConversion : ConvertOpToLLVMPattern<ttg::WarpIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ttg::WarpIdOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    Value tid = readCSR(op.getLoc(), rewriter, triton::vortex::ThreadIdxX);
    int threads = ttg::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    rewriter.replaceOp(op, b.udiv(tid, b.i32_val(threads)));
    return success();
  }
};

struct MakeRangeOpConversion : ConvertOpToLLVMPattern<triton::MakeRangeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getType();
    auto layout = ttg::toLinearLayout(type).removeZeroBasesAlongDim(
        rewriter.getStringAttr("register"));
    auto [lane, warp] = getLaneAndWarpId(rewriter, loc);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> values;
    for (unsigned reg = 0; reg < ttg::getUniqueElemsPerThread(type); ++reg) {
      SmallVector<std::pair<StringAttr, Value>> inputs;
      for (StringAttr dim : layout.getInDimNames()) {
        Value value = b.i32_val(0);
        if (dim.getValue() == "register")
          value = b.i32_val(reg);
        else if (dim.getValue() == "lane")
          value = lane;
        else if (dim.getValue() == "warp")
          value = warp;
        inputs.emplace_back(dim, value);
      }
      auto index = applyLinearLayout(loc, rewriter, layout, inputs);
      values.push_back(b.add(index.front().second, b.i32_val(op.getStart())));
    }
    rewriter.replaceOp(op, packUniqueTensorElements(loc, getTypeConverter(),
                                                    values, rewriter, type));
    return success();
  }
};

struct FuncOpConversion : ConvertOpToLLVMPattern<triton::FuncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converted =
        convertFuncOpToLLVMFuncOp(op, rewriter, *getTypeConverter());
    if (failed(converted))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpConversion : ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

Value guardedLoad(Location loc, Operation *op, Value pointer, Type type,
                  Value mask, Value other, bool isVolatile,
                  ConversionPatternRewriter &rewriter) {
  unsigned alignment = type.getIntOrFloatBitWidth() / 8;
  if (!mask)
    return LLVM::LoadOp::create(rewriter, loc, type, pointer, alignment,
                                isVolatile);
  Block *head = rewriter.getInsertionBlock();
  Block *merge = rewriter.splitBlock(head, Block::iterator(op));
  Value result = merge->addArgument(type, loc);
  Block *read = rewriter.createBlock(merge);
  Value loaded =
      LLVM::LoadOp::create(rewriter, loc, type, pointer, alignment, isVolatile);
  LLVM::BrOp::create(rewriter, loc, loaded, merge);
  rewriter.setInsertionPointToEnd(head);
  LLVM::CondBrOp::create(rewriter, loc, mask, read, ValueRange{}, merge,
                         ValueRange{other});
  rewriter.setInsertionPoint(op);
  return result;
}

struct LoadOpConversion : ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getCachePolicyAttr())
      return rewriter.notifyMatchFailure(
          op, "cache policy lowering is not implemented");
    auto loc = op.getLoc();
    auto pointers = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    SmallVector<Value> masks, others, values;
    if (adaptor.getMask())
      masks = unpackLLElements(loc, adaptor.getMask(), rewriter);
    if (adaptor.getOther())
      others = unpackLLElements(loc, adaptor.getOther(), rewriter);
    Type type =
        getTypeConverter()->convertType(getElementTypeOrSelf(op.getType()));
    for (unsigned i = 0; i < pointers.size(); ++i) {
      Value other = others.empty() ? LLVM::UndefOp::create(rewriter, loc, type)
                                   : others[i];
      values.push_back(guardedLoad(loc, op, pointers[i], type,
                                   masks.empty() ? Value() : masks[i], other,
                                   op.getIsVolatile(), rewriter));
    }
    rewriter.replaceOp(op,
                       packUniqueTensorElements(loc, getTypeConverter(), values,
                                                rewriter, op.getType()));
    return success();
  }
};

struct StoreOpConversion : ConvertOpToLLVMPattern<triton::StoreOp> {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const triton::vortex::TargetInfo &targetInfo,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}
  const triton::vortex::TargetInfo &targetInfo;
  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getCachePolicyAttr())
      return rewriter.notifyMatchFailure(
          op, "cache policy lowering is not implemented");
    auto loc = op.getLoc();
    auto pointers = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto values = unpackLLElements(loc, adaptor.getValue(), rewriter);
    SmallVector<Value> masks;
    if (adaptor.getMask())
      masks = unpackLLElements(loc, adaptor.getMask(), rewriter);
    Value threadPred = ttg::emitRedundantThreadPredicate(
        getFreeVariableMasks(op.getPtr().getType()), rewriter, loc, targetInfo);
    for (unsigned i = 0; i < pointers.size(); ++i) {
      Value mask = ttg::maybeAnd(rewriter, loc, threadPred,
                                masks.empty() ? Value() : masks[i]);
      unsigned alignment = values[i].getType().getIntOrFloatBitWidth() / 8;
      if (!mask) {
        LLVM::StoreOp::create(rewriter, loc, values[i], pointers[i], alignment);
        continue;
      }
      Block *head = rewriter.getInsertionBlock();
      Block *merge = rewriter.splitBlock(head, Block::iterator(op));
      Block *write = rewriter.createBlock(merge);
      LLVM::StoreOp::create(rewriter, loc, values[i], pointers[i], alignment);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, merge);
      rewriter.setInsertionPointToEnd(head);
      LLVM::CondBrOp::create(rewriter, loc, mask, write, merge);
      rewriter.setInsertionPoint(op);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicRMWOpConversion : ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const triton::vortex::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getElementTypeOrSelf(op.getType());
    if (!type.isInteger(32) || op.getAtomicRmwOp() != RMWOp::ADD ||
        op.getSem() != MemSemantic::RELAXED)
      return rewriter.notifyMatchFailure(
          op, "Vortex atomics currently require relaxed i32 addition");

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto pointers = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto values = unpackLLElements(loc, adaptor.getVal(), rewriter);
    SmallVector<Value> masks, results;
    if (adaptor.getMask())
      masks = unpackLLElements(loc, adaptor.getMask(), rewriter);
    // Suppress duplicate layout owners, not distinct elements that happen to
    // address the same word: those must each perform their atomic update.
    Value threadPred = ttg::emitRedundantThreadPredicate(
        getFreeVariableMasks(op.getPtr().getType()), rewriter, loc, targetInfo);
    for (unsigned i = 0; i < pointers.size(); ++i) {
      Value mask = ttg::maybeAnd(rewriter, loc, threadPred,
                                masks.empty() ? Value() : masks[i]);
      if (!mask)
        mask = b.true_val();
      Value inactive = b.undef(type);
      Block *head = rewriter.getInsertionBlock();
      Block *merge = rewriter.splitBlock(head, Block::iterator(op));
      Value result = merge->addArgument(type, loc);
      Block *update = rewriter.createBlock(merge);
      // Default LLVM system scope also satisfies Triton's GPU/CTA scopes.
      // The SDK lowers this to RISC-V AMOADD.W, without acquire/release bits.
      Value old = LLVM::AtomicRMWOp::create(
          rewriter, loc, LLVM::AtomicBinOp::add, pointers[i], values[i],
          LLVM::AtomicOrdering::monotonic);
      LLVM::BrOp::create(rewriter, loc, old, merge);
      rewriter.setInsertionPointToEnd(head);
      LLVM::CondBrOp::create(rewriter, loc, mask, update, ValueRange{}, merge,
                            ValueRange{inactive});
      rewriter.setInsertionPoint(op);
      results.push_back(result);
    }
    if (op->use_empty()) {
      rewriter.eraseOp(op);
    } else {
      // Replicate canonical owners' old values for redundant lanes/warps.
      if (!threadPred)
        threadPred = b.true_val();
      finalizeAtomicResults(op, rewriter, results, type, b, threadPred,
                            targetInfo, getTypeConverter());
    }
    return success();
  }

  const triton::vortex::TargetInfo &targetInfo;
};

struct BarrierOpConversion : ConvertOpToLLVMPattern<ttg::BarrierOp> {
  BarrierOpConversion(LLVMTypeConverter &converter,
                      const triton::vortex::TargetInfo &targetInfo,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}
  LogicalResult
  matchAndRewrite(ttg::BarrierOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    targetInfo.barrier(op.getLoc(), rewriter, op.getAddrSpace());
    rewriter.eraseOp(op);
    return success();
  }
  const triton::vortex::TargetInfo &targetInfo;
};

struct ConvertTritonGPUToVortexLLVM
    : triton::vortex::impl::ConvertTritonGPUToVortexLLVMBase<
          ConvertTritonGPUToVortexLLVM> {
  void populateConversionPatterns(VortexTypeConverter &converter,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfo,
                                  const triton::vortex::TargetInfo &targetInfo) {
    auto benefit = PatternBenefit(patternBenefitPrioritizeOverLLVMConversions);
    populateElementwiseOpToLLVMPatterns(converter, patterns, axisInfo, benefit);
    patterns.add<ttg::ElementwiseOpConversion<arith::AddFOp, LLVM::FAddOp>>(
        converter, axisInfo, benefit);
    populateViewOpToLLVMPatterns(converter, patterns, benefit);
    populateReduceOpToLLVMPatterns(converter, patterns, targetInfo, benefit);
    populateScanOpToLLVMPatterns(converter, patterns, targetInfo, benefit);
    populateConvertLayoutOpToLLVMPatterns(converter, targetInfo, patterns, benefit);
    patterns
        .add<GetProgramIdOpConversion, GetNumProgramsOpConversion,
             ThreadIdOpConversion, WarpIdOpConversion, MakeRangeOpConversion,
             LoadOpConversion, ReturnOpConversion>(converter, benefit);
    patterns.add<StoreOpConversion, AtomicRMWOpConversion, BarrierOpConversion>(
        converter, targetInfo, benefit);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    auto targetName = mod->getAttrOfType<StringAttr>("ttg.target");
    if (!targetName || targetName.getValue() != "vortex:32" ||
        ttg::TritonGPUDialect::getNumCTAs(mod) != 1) {
      mod.emitError("expected vortex:32 with one CTA per program");
      signalPassFailure();
      return;
    }
    WalkResult supported = mod.walk([&](Operation *op) {
      if (isa<triton::ScanOp>(op) &&
          llvm::any_of(op->getResultTypes(), [](Type type) {
            return !getElementTypeOrSelf(type).isInteger(32);
          })) {
        op->emitError("Vortex scans currently require i32 elements");
        return WalkResult::interrupt();
      }
      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        if (llvm::any_of(reduce.getElementTypes(),
                         [](Type type) { return !type.isInteger(32); })) {
          op->emitError("Vortex reductions currently require i32 elements");
          return WalkResult::interrupt();
        }
      }
      for (Type type : op->getResultTypes()) {
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
          Type element = tensor.getElementType();
          if (auto ptr = dyn_cast<triton::PointerType>(element))
            element = ptr.getPointeeType();
          // Generic view/layout patterns handle rank-two tiles and the sliced
          // rank-one layouts used to construct their row/column coordinates.
          Attribute layout = tensor.getEncoding();
          while (auto slice = dyn_cast<ttg::SliceEncodingAttr>(layout))
            layout = slice.getParent();
          auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout);
          if (tensor.getRank() < 1 || tensor.getRank() > 2 || !blocked ||
              blocked.getSizePerThread().size() > 2 ||
              !(isa<IntegerType>(element) || element.isF32())) {
            op->emitError(
                "Vortex requires rank-one/two integer or f32 tensors with "
                "blocked layouts or slices of rank-one/two blocked layouts");
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (supported.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    triton::vortex::TargetInfo targetInfo(
        ttg::TritonGPUDialect::getThreadsPerWarp(mod));
    ModuleAllocation allocation(mod);
    ttg::attachAllocationSizeAndOffsetAttr(mod, allocation);
    ModuleMembarAnalysis membar(allocation);
    membar.run();
    VortexTypeConverter converter(ctx);
    RewritePatternSet functions(ctx);
    functions.add<FuncOpConversion>(converter);
    ConversionTarget functionTarget(*ctx);
    functionTarget.markUnknownOpDynamicallyLegal(
        [](Operation *) { return true; });
    functionTarget.addIllegalOp<triton::FuncOp>();
    if (failed(applyPartialConversion(mod, functionTarget,
                                      std::move(functions)))) {
      signalPassFailure();
      return;
    }
    ModuleAxisInfoAnalysis axisInfo(mod);
    RewritePatternSet patterns(ctx);
    populateConversionPatterns(converter, patterns, axisInfo, targetInfo);
    // Triton's generic layout conversions resolve scratch through this symbol.
    // Vortex provides the address through a per-CTA CSR instead of a linker symbol.
    OpBuilder globals(mod.getBodyRegion());
    auto smem = LLVM::GlobalOp::create(
        globals, mod.getLoc(), LLVM::LLVMArrayType::get(globals.getI8Type(), 0),
        false, LLVM::Linkage::External, "global_smem", Attribute(), 16, 0);
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    if (failed(applyFullConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(ctx);
    mod.walk([&](LLVM::AddressOfOp op) {
      if (op.getGlobalName() != "global_smem")
        return;
      rewriter.setInsertionPoint(op);
      auto address = readCSR(op.getLoc(), rewriter,
                             triton::vortex::CTALocalMemory);
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, op.getType(), address);
    });
    smem.erase();
  }
};

} // namespace

// External backends cannot edit triton-opt's central registration header.
static mlir::PassRegistration<ConvertTritonGPUToVortexLLVM> registerVortexPass;
