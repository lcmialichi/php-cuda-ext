#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif
    int cuda_wrapper_get_device_count();
    int cuda_wrapper_get_device_properties(int device_id, char *name, size_t name_size,
                                           int *major, int *minor, size_t *total_mem);
    int cuda_wrapper_set_device(int device_id);
    int cuda_wrapper_get_current_device();
    int cuda_wrapper_get_memory_info(size_t *free_mem, size_t *total_mem);
    int cuda_wrapper_device_reset();
    int cuda_wrapper_get_driver_version();
    int cuda_wrapper_get_runtime_version();
    int cuda_wrapper_synchronize();
    int cuda_wrapper_get_peer_access(int device1, int device2);

    const char *cuda_wrapper_get_error_string(int error);
    const char *cuda_wrapper_get_error_type(int error);
    int *cuda_wrapper_error();

#ifdef __cplusplus
}
#endif

#endif