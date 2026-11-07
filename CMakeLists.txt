include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonVortex triton_vortex.cc LINK_LIBS TritonVortexGPUToLLVM)
  target_link_libraries(TritonVortex PRIVATE Python3::Module nanobind-static)
endif()
