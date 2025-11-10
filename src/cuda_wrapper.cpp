#include "cuda_wrapper.h"
#include <cuda_runtime.h>
#include <cstring>

extern "C"
{
    int cuda_wrapper_get_device_count()
    {
        int count;
        cudaError_t error = cudaGetDeviceCount(&count);
        return (error == cudaSuccess) ? count : -1;
    }

    int cuda_wrapper_get_device_properties(int device_id, char *name, size_t name_size,
                                           int *major, int *minor, size_t *total_mem)
    {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, device_id);

        if (error != cudaSuccess)
        {
            return 0;
        }

        strncpy(name, prop.name, name_size - 1);
        name[name_size - 1] = '\0';
        *major = prop.major;
        *minor = prop.minor;
        *total_mem = prop.totalGlobalMem;

        return 1;
    }

    int cuda_wrapper_set_device(int device_id)
    {
        cudaError_t error = cudaSetDevice(device_id);
        return (error == cudaSuccess) ? 1 : 0;
    }

    int cuda_wrapper_get_current_device()
    {
        int device;
        cudaError_t error = cudaGetDevice(&device);
        return (error == cudaSuccess) ? device : -1;
    }

    int cuda_wrapper_get_memory_info(size_t *free_mem, size_t *total_mem)
    {
        int status = cudaMemGetInfo(free_mem, total_mem);
        if (status == cudaSuccess)
        {
            return 1;
        }

        return 0;
    }

    int cuda_wrapper_device_reset()
    {
        int status = cudaDeviceReset();
        if (status == cudaSuccess)
        {
            return 1;
        }

        return 0;
    }

    int cuda_wrapper_get_driver_version()
    {
        int version;
        cudaError_t error = cudaDriverGetVersion(&version);
        return (error == cudaSuccess) ? version : -1;
    }

    int cuda_wrapper_get_runtime_version()
    {
        int version;
        cudaError_t error = cudaRuntimeGetVersion(&version);
        return (error == cudaSuccess) ? version : -1;
    }

    int cuda_wrapper_synchronize()
    {
        cudaError_t error = cudaDeviceSynchronize();
        return (error != cudaSuccess) ? 0 : 1;
    }

    int cuda_wrapper_get_peer_access(int device1, int device2)
    {
        int can_access;
        cudaError_t error = cudaDeviceCanAccessPeer(&can_access, device1, device2);
        return (error == cudaSuccess) ? can_access : -1;
    }

    int cuda_wrapper_error()
    {
        cudaError_t error = cudaGetLastError();
        return (int)error;
    }

    const char *cuda_wrapper_get_error_string(int error)
    {
        return cudaGetErrorString((cudaError_t)error);
    }

    const char *cuda_wrapper_get_error_type(int error)
    {
        return cuda_error_type((cudaError_t)error);
    }

    const char *cuda_error_type(cudaError_t error)
    {
        switch (error)
        {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        default:
            return "cudaErrorUnknown";
        }
    }

}