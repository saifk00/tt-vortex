add_triton_library(TritonVortexGPUToLLVM
  TritonGPUToLLVM.cpp

  DEPENDS
  TritonVortexGPUConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonGPUToLLVM
  TritonAnalysis
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
)
