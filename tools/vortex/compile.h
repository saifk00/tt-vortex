#ifndef TTVORTEX_{kernel_name}_H
#define TTVORTEX_{kernel_name}_H

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {{
#endif

vx_result_t load_{kernel_name}(vx_device_h device);
vx_result_t unload_{kernel_name}(void);
// Device pointer arguments are uint64_t addresses from vx_buffer_address.
// Launch is asynchronous; finish the queue before unloading or closing device.
vx_result_t {kernel_name}(vx_device_h device, vx_queue_h queue
#if {num_args} > 2
    , {signature}
#endif
);

// Opt in for a single specialization. Keep hashed names to use several variants.
#ifdef TTVORTEX_AOT_SHORT_NAMES
#ifdef TTVORTEX_SHORT_NAME_{triton_kernel_name}
#error "Short names already selected for another specialization of this kernel"
#endif
#define TTVORTEX_SHORT_NAME_{triton_kernel_name}
// Assembly labels preserve unique linker names without hiding functions behind macros.
vx_result_t load_{triton_kernel_name}(vx_device_h device) __asm__("load_{kernel_name}");
vx_result_t unload_{triton_kernel_name}(void) __asm__("unload_{kernel_name}");
vx_result_t {triton_kernel_name}(vx_device_h device, vx_queue_h queue
#if {num_args} > 2
    , {signature}
#endif
) __asm__("{kernel_name}");
#endif

#ifdef __cplusplus
}}
#endif

#endif
