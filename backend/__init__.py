"""External Vortex backend for a prebuilt Triton compiler."""

# Triton discovers our compiler/driver during its own import. Complete that
# discovery before a user imports either submodule directly, or Triton would
# inspect a partially initialized module with no backend class yet.
import triton as _triton
