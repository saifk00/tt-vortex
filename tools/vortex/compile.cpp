#include <stdint.h>
#include <stddef.h>
#include <cstring>
#include <array>
#include <vortex2.h>

// Short source names bind to the same specialization-specific linker symbols.
#ifndef TTVORTEX_AOT_SHORT_NAMES
#define {triton_kernel_name} {kernel_name}
#define load_{triton_kernel_name} load_{kernel_name}
#define unload_{triton_kernel_name} unload_{kernel_name}
#endif

extern "C" vx_result_t load_{triton_kernel_name}(vx_device_h device) __asm__("load_{kernel_name}");
extern "C" vx_result_t unload_{triton_kernel_name}(void) __asm__("unload_{kernel_name}");
extern "C" vx_result_t {triton_kernel_name}(vx_device_h device, vx_queue_h queue
#if {num_args} > 2
    , {signature}
#endif
) __asm__("{kernel_name}");

static const unsigned char {kernel_name}_vxbin[{bin_size}] = {{ {bin_data} }};
static vx_device_h {kernel_name}_device;
static vx_module_h {kernel_name}_module;
static vx_kernel_h {kernel_name}_kernel;

extern "C" vx_result_t load_{triton_kernel_name}(vx_device_h device) {{
    if (!device)
        return VX_ERR_INVALID_HANDLE;
    if ({kernel_name}_module)
        return device == {kernel_name}_device ? VX_SUCCESS : VX_ERR_INVALID_VALUE;
    uint64_t threads, warps;
    vx_result_t result = vx_device_query(device, VX_CAPS_NUM_THREADS, &threads);
    if (result != VX_SUCCESS)
        return result;
    result = vx_device_query(device, VX_CAPS_NUM_WARPS, &warps);
    if (result != VX_SUCCESS)
        return result;
    if (threads != {warp_size} || warps < {num_warps})
        return VX_ERR_NOT_SUPPORTED;
    result = vx_module_load_bytes(device, {kernel_name}_vxbin,
                                  sizeof({kernel_name}_vxbin), &{kernel_name}_module);
    if (result != VX_SUCCESS)
        return result;
    result = vx_module_get_kernel({kernel_name}_module, "main", &{kernel_name}_kernel);
    if (result != VX_SUCCESS) {{
        vx_module_release({kernel_name}_module);
        {kernel_name}_module = NULL;
        return result;
    }}
    {kernel_name}_device = device;
    return VX_SUCCESS;
}}

extern "C" vx_result_t unload_{triton_kernel_name}(void) {{
    vx_result_t result;
    if ({kernel_name}_kernel) {{
        result = vx_kernel_release({kernel_name}_kernel);
        if (result != VX_SUCCESS)
            return result;
        {kernel_name}_kernel = NULL;
    }}
    if ({kernel_name}_module) {{
        result = vx_module_release({kernel_name}_module);
        if (result != VX_SUCCESS)
            return result;
        {kernel_name}_module = NULL;
    }}
    {kernel_name}_device = NULL;
    return VX_SUCCESS;
}}

// The upstream CLI appends two scratch arguments; this backend uses neither.
// Preserve argument types until packing, then hand the runtime one byte blob.
template <typename... Args>
static auto pack_arguments(const Args*... args) {{
    struct Blob {{
        std::array<unsigned char, sizeof...(Args) * 8> data{{}};
        size_t size = 0;
    }} blob;
    size_t index = 0;
    auto append = [&](const auto* arg) {{
        constexpr size_t size = sizeof(*arg);
        static_assert(size == 1 || size == 2 || size == 4 || size == 8,
                      "unsupported Vortex argument size");
        if (index++ >= sizeof...(Args) - 2)
            return;
        blob.size = (blob.size + size - 1) & ~(size - 1);
        std::memcpy(blob.data.data() + blob.size, arg, size);
        blob.size += size;
    }};
    (append(args), ...);
    blob.size = (blob.size + 7) & ~size_t(7);
    return blob;
}}

/*
{kernel_docstring}
*/
extern "C" vx_result_t {triton_kernel_name}(vx_device_h device, vx_queue_h queue
#if {num_args} > 2
    , {signature}
#endif
) {{
    const uint32_t gx = {gridX};
    const uint32_t gy = {gridY};
    const uint32_t gz = {gridZ};
    if (!gx || !gy || !gz)
        return VX_SUCCESS;
    vx_result_t result = load_{triton_kernel_name}(device);
    if (result != VX_SUCCESS)
        return result;

    uint64_t global_scratch = 0, profile_scratch = 0;
    auto blob = pack_arguments({arg_pointers});
    vx_launch_info_t launch{{}};
    launch.struct_size = sizeof(launch);
    launch.kernel = {kernel_name}_kernel;
    launch.args_host = blob.data.data();
    launch.args_size = blob.size;
    launch.ndim = 3;
    launch.grid_dim[0] = gx;
    launch.grid_dim[1] = gy;
    launch.grid_dim[2] = gz;
    launch.block_dim[0] = {num_warps} * {warp_size};
    launch.block_dim[1] = 1;
    launch.block_dim[2] = 1;
    launch.lmem_size = {shared};
    return vx_enqueue_launch(queue, &launch, 0, NULL, NULL);
}}
