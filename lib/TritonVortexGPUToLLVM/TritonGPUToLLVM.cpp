#include "TritonVortexGPUToLLVM/CSR.h"
#include "TritonVortexGPUToLLVM/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/AxisInfo.h"
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

Value readCSR(Location loc, ConversionPatternRewriter &rewriter, uint32_t csr) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return LLVM::InlineAsmOp::create(
             rewriter, loc, rewriter.getI32Type(), ValueRange{b.i32_val(csr)},
             "csrr $0, $1", "=r,i", false, false, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                       LLVM::AsmDialect::AD_ATT),
             ArrayAttr())
      .getResult(0);
}

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
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
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
    for (unsigned i = 0; i < pointers.size(); ++i) {
      unsigned alignment = values[i].getType().getIntOrFloatBitWidth() / 8;
      if (masks.empty()) {
        LLVM::StoreOp::create(rewriter, loc, values[i], pointers[i], alignment);
        continue;
      }
      Block *head = rewriter.getInsertionBlock();
      Block *merge = rewriter.splitBlock(head, Block::iterator(op));
      Block *write = rewriter.createBlock(merge);
      LLVM::StoreOp::create(rewriter, loc, values[i], pointers[i], alignment);
      LLVM::BrOp::create(rewriter, loc, ValueRange{}, merge);
      rewriter.setInsertionPointToEnd(head);
      LLVM::CondBrOp::create(rewriter, loc, masks[i], write, merge);
      rewriter.setInsertionPoint(op);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTritonGPUToVortexLLVM
    : triton::vortex::impl::ConvertTritonGPUToVortexLLVMBase<
          ConvertTritonGPUToVortexLLVM> {
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
      for (Type type : op->getResultTypes()) {
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
          Type element = tensor.getElementType();
          if (auto ptr = dyn_cast<triton::PointerType>(element))
            element = ptr.getPointeeType();
          if (tensor.getRank() != 1 ||
              !isa<ttg::BlockedEncodingAttr>(tensor.getEncoding()) ||
              !isa<IntegerType>(element)) {
            op->emitError(
                "Vortex currently supports rank-one blocked integer tensors");
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
    auto benefit = PatternBenefit(patternBenefitPrioritizeOverLLVMConversions);
    populateElementwiseOpToLLVMPatterns(converter, patterns, axisInfo, benefit);
    populateViewOpToLLVMPatterns(converter, patterns, benefit);
    patterns
        .add<GetProgramIdOpConversion, GetNumProgramsOpConversion,
             ThreadIdOpConversion, WarpIdOpConversion, MakeRangeOpConversion,
             LoadOpConversion, StoreOpConversion, ReturnOpConversion>(converter,
                                                                      benefit);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    if (failed(applyFullConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// External backends cannot edit triton-opt's central registration header.
static mlir::PassRegistration<ConvertTritonGPUToVortexLLVM> registerVortexPass;
