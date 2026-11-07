#include "TritonVortexGPUToLLVM/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <stdexcept>

namespace py = nanobind;

// Device addresses occupy uint64_t slots in the host runtime's argument blob,
// even though pointers in the RV32 kernel are 32 bits.
static unsigned addLaunchWrapper(llvm::Module &module, const std::string &name) {
  auto *body = module.getFunction(name);
  if (!body || !body->getReturnType()->isVoidTy())
    throw std::runtime_error("Vortex requires a void kernel entry");
  for (auto &arg : body->args()) {
    auto *ty = arg.getType();
    if (!ty->isPointerTy() && !ty->isIntegerTy(8) && !ty->isIntegerTy(16) &&
        !ty->isIntegerTy(32) && !ty->isIntegerTy(64))
      throw std::runtime_error("Vortex launch arguments must be pointers or i8/i16/i32/i64 scalars");
  }
  body->setName("__ttvortex_body_" + name);
  body->setLinkage(llvm::GlobalValue::InternalLinkage);
  body->addFnAttr(llvm::Attribute::AlwaysInline);
  auto &ctx = module.getContext();
  llvm::IRBuilder<> b(ctx);
  auto *ptrTy = b.getPtrTy();
  auto *entry = llvm::Function::Create(
      llvm::FunctionType::get(b.getVoidTy(), {ptrTy}, false),
      llvm::GlobalValue::ExternalLinkage, "kernel_main", module);
  b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", entry));
  auto uniformPtr = module.getOrInsertFunction(
      "llvm.riscv.vx.uniform.p0.p0", llvm::FunctionType::get(ptrTy, {ptrTy}, false));
  auto uniformInt = module.getOrInsertFunction(
      "llvm.riscv.vx.uniform.i32.i32",
      llvm::FunctionType::get(b.getInt32Ty(), {b.getInt32Ty()}, false));
  auto *args = b.CreateCall(uniformPtr, {entry->getArg(0)}, "args");
  llvm::SmallVector<llvm::Value *> values;
  unsigned offset = 0;
  for (auto &arg : body->args()) {
    auto *ty = arg.getType();
    unsigned size = ty->isPointerTy() ? 8 : ty->getIntegerBitWidth() / 8;
    offset = llvm::alignTo(offset, size);
    auto *slot = b.CreateConstGEP1_32(b.getInt8Ty(), args, offset);
    auto *storageTy = ty->isPointerTy() ? b.getInt64Ty() : ty;
    llvm::Value *value = b.CreateAlignedLoad(storageTy, slot, llvm::Align(size));
    if (ty->isPointerTy()) {
      value = b.CreateIntToPtr(b.CreateTrunc(value, b.getInt32Ty()), ty);
      value = b.CreateCall(uniformPtr, {value});
    } else if (ty->isIntegerTy(32)) {
      value = b.CreateCall(uniformInt, {value});
    }
    values.push_back(value);
    offset += size;
  }
  b.CreateCall(body, values);
  b.CreateRetVoid();
  auto *annotation = llvm::ConstantStruct::getAnon(
      {entry, b.CreateGlobalString("vortex.kernel"),
       b.CreateGlobalString("ttvortex"), b.getInt32(1),
       llvm::ConstantPointerNull::get(ptrTy)});
  auto *annotationsTy = llvm::ArrayType::get(annotation->getType(), 1);
  auto *annotations = new llvm::GlobalVariable(
      module, annotationsTy, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(annotationsTy, {annotation}), "llvm.global.annotations");
  annotations->setSection("llvm.metadata");
  llvm::appendToUsed(module, {entry});
  return llvm::alignTo(offset, 8);
}

static void init_passes(py::module_ &m) {
  ADD_PASS_WRAPPER_0("add_to_llvmir",
                     mlir::triton::vortex::createConvertTritonGPUToVortexLLVM);
}

void init_triton_vortex(py::module_ &m) {
  auto passes = m.def_submodule("passes");
  auto ttgpuir = passes.def_submodule("ttgpuir");
  init_passes(ttgpuir);
  m.def("add_launch_wrapper", &addLaunchWrapper);
  m.def("set_target", [](llvm::Module &module) {
    module.setTargetTriple(llvm::Triple("riscv32-unknown-elf"));
    module.setDataLayout("e-m:e-p:32:32-i64:64-n32-S128");
  });
}
