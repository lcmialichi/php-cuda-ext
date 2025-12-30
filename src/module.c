#include "module.h"
#include "php.h"
#include "ca_struct.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "module_arginfo.h"
#include "kernel_types.h"
#include "tensor.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "ext/standard/php_standard.h"

#define MAX_KERNEL_ERROR_LEN 1024
#define ERROR_CHECK_INTERVAL 50
#define ASYNC_OP_TIMEOUT_MS 30000
#define INITIAL_STREAM_POOL_SIZE 4
#define MAX_STREAM_POOL_SIZE 25
#define MIN_KERNEL_SIZE_FOR_ASYNC 256

extern zend_class_entry *cuda_array_ce;

zend_class_entry *cuda_module_ce;
static zend_object_handlers module_handlers;
static pthread_mutex_t g_cuda_global_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static zend_bool g_cuda_initialized = 0;
static CUdevice g_primary_device = 0;

static zend_object *module_create_object(zend_class_entry *class_type);
static void module_free_object(zend_object *object);
static zend_bool module_initialize_cuda_context(cuda_module_object *module);
static void module_cleanup_cuda_resources(cuda_module_object *module);
static const char *module_get_cuda_error_string(CUresult result);
static double module_get_current_time_ms(void);
static const char *module_dtype_to_string(dtype_t dtype);
static CUmodule module_get_or_load_module(cuda_module_object *module, zend_string *kernel_name);
static zend_bool module_validate_launch_config(cuda_module_object *module, int grid[3], int block[3]);
static zend_bool module_prepare_cuda_arguments(cuda_kernel_data *kernel, zval *args, int argc,
                                               void ***cuda_args_ptr, void ***temp_buffers_ptr,
                                               int *temp_buffers_count_ptr);
static zend_bool module_execute_cuda_kernel(cuda_module_object *module,
                                            cuda_kernel_data *kernel,
                                            int grid[3], int block[3],
                                            void **cuda_args, int argc,
                                            CUstream stream);
static void module_check_cuda_error(cuda_module_object *module, CUresult result, const char *context);
static int module_create_async_operation(cuda_module_object *module,
                                         zend_string *kernel_name,
                                         void **cuda_args,
                                         void **temp_buffers,
                                         int temp_buffers_count,
                                         int grid[3],
                                         int block[3],
                                         int argc);
static void module_cleanup_async_operation_by_id(cuda_module_object *module, int op_id);
static void module_cleanup_all_async_operations(cuda_module_object *module);
static void module_cleanup_timeout_operations(cuda_module_object *module);
static zend_bool module_validate_tensor_access(tensor_t *tensor, int total_threads);
static void module_log_error(const char *format, ...);
static CUstream module_get_stream_from_pool(cuda_module_object *module);
static void module_return_stream_to_pool(cuda_module_object *module, CUstream stream);
static void module_initialize_stream_pool(cuda_module_object *module);
static void module_destroy_stream_pool(cuda_module_object *module);
static zend_bool module_initialize_global_cuda(cuda_module_object *module);
static void module_prepare_launch_config(zval *config_zv, int grid[3], int block[3]);
static void module_cleanup_args_and_buffers(zval *args, void **cuda_args,
                                            void **temp_buffers, int temp_buffers_count);
static zend_bool module_should_use_async(int grid[3], int block[3]);
static zend_bool module_validate_async_operation_count(cuda_module_object *module);

static void module_log_error(const char *format, ...)
{
    va_list args;
    char buffer[512];

    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    php_error_docref(NULL, E_WARNING, "[CUDA Module] %s", buffer);
}

static void module_check_cuda_error(cuda_module_object *module, CUresult result, const char *context)
{
    if (result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "CUDA error in %s: %s (code: %d)",
                                context,
                                module_get_cuda_error_string(result),
                                result);
    }
}

static const char *module_get_cuda_error_string(CUresult result)
{
    switch (result)
    {
    case CUDA_SUCCESS:
        return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
        return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
        return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
        return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:
        return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_NO_DEVICE:
        return "CUDA_ERROR_NO_DEVICE";
    case CUDA_ERROR_INVALID_DEVICE:
        return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_IMAGE:
        return "CUDA_ERROR_INVALID_IMAGE";
    case CUDA_ERROR_INVALID_CONTEXT:
        return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case CUDA_ERROR_MAP_FAILED:
        return "CUDA_ERROR_MAP_FAILED";
    case CUDA_ERROR_UNMAP_FAILED:
        return "CUDA_ERROR_UNMAP_FAILED";
    case CUDA_ERROR_ARRAY_IS_MAPPED:
        return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case CUDA_ERROR_ALREADY_MAPPED:
        return "CUDA_ERROR_ALREADY_MAPPED";
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
        return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED:
        return "CUDA_ERROR_ALREADY_ACQUIRED";
    case CUDA_ERROR_NOT_MAPPED:
        return "CUDA_ERROR_NOT_MAPPED";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
        return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
        return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case CUDA_ERROR_ECC_UNCORRECTABLE:
        return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
        return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
        return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case CUDA_ERROR_INVALID_PTX:
        return "CUDA_ERROR_INVALID_PTX";
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
        return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
        return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
    case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
        return "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";
    case CUDA_ERROR_JIT_COMPILATION_DISABLED:
        return "CUDA_ERROR_JIT_COMPILATION_DISABLED";
    case CUDA_ERROR_INVALID_SOURCE:
        return "CUDA_ERROR_INVALID_SOURCE";
    case CUDA_ERROR_FILE_NOT_FOUND:
        return "CUDA_ERROR_FILE_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
        return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case CUDA_ERROR_OPERATING_SYSTEM:
        return "CUDA_ERROR_OPERATING_SYSTEM";
    case CUDA_ERROR_INVALID_HANDLE:
        return "CUDA_ERROR_INVALID_HANDLE";
    case CUDA_ERROR_ILLEGAL_STATE:
        return "CUDA_ERROR_ILLEGAL_STATE";
    case CUDA_ERROR_NOT_FOUND:
        return "CUDA_ERROR_NOT_FOUND";
    case CUDA_ERROR_NOT_READY:
        return "CUDA_ERROR_NOT_READY";
    case CUDA_ERROR_ILLEGAL_ADDRESS:
        return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case CUDA_ERROR_LAUNCH_TIMEOUT:
        return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
        return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
        return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
        return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    case CUDA_ERROR_ASSERT:
        return "CUDA_ERROR_ASSERT";
    case CUDA_ERROR_TOO_MANY_PEERS:
        return "CUDA_ERROR_TOO_MANY_PEERS";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
        return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
        return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
        return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    case CUDA_ERROR_MISALIGNED_ADDRESS:
        return "CUDA_ERROR_MISALIGNED_ADDRESS";
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
        return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    case CUDA_ERROR_INVALID_PC:
        return "CUDA_ERROR_INVALID_PC";
    case CUDA_ERROR_LAUNCH_FAILED:
        return "CUDA_ERROR_LAUNCH_FAILED";
    case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
        return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
    case CUDA_ERROR_NOT_PERMITTED:
        return "CUDA_ERROR_NOT_PERMITTED";
    case CUDA_ERROR_NOT_SUPPORTED:
        return "CUDA_ERROR_NOT_SUPPORTED";
    case CUDA_ERROR_SYSTEM_NOT_READY:
        return "CUDA_ERROR_SYSTEM_NOT_READY";
    case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
        return "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";
    case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
        return "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE";
    case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
        return "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
    case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
        return "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
    case CUDA_ERROR_TIMEOUT:
        return "CUDA_ERROR_TIMEOUT";
    case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
        return "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
    case CUDA_ERROR_EXTERNAL_DEVICE:
        return "CUDA_ERROR_EXTERNAL_DEVICE";
    case CUDA_ERROR_UNKNOWN:
        return "CUDA_ERROR_UNKNOWN";
    default:
        return "Unknown CUDA error";
    }
}

static double module_get_current_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

static const char *module_dtype_to_string(dtype_t dtype)
{
    switch (dtype)
    {
    case FLOAT32:
        return "float32";
    case FLOAT64:
        return "float64";
    case INT32:
        return "int32";
    case INT64:
        return "int64";
    case BOOL:
        return "bool";
    case LIST:
        return "array";
    default:
        return "unknown";
    }
}

static zend_bool module_validate_tensor_access(tensor_t *tensor, int total_threads)
{
    if (!tensor)
        return 0;

    size_t tensor_elements = 1;
    for (int i = 0; i < tensor->ndims; i++)
    {
        tensor_elements *= tensor->shape[i];
    }

    if (total_threads > tensor_elements)
    {
        module_log_error("Potential out-of-bounds access: kernel configured for %d threads, "
                         "but tensor has only %zu elements",
                         total_threads, tensor_elements);
        return 0;
    }

    return 1;
}

static void module_prepare_launch_config(zval *config_zv, int grid[3], int block[3])
{
    grid[0] = grid[1] = grid[2] = 1;
    block[0] = block[1] = block[2] = 1;

    if (config_zv && Z_TYPE_P(config_zv) == IS_ARRAY)
    {
        zval *grid_zv = zend_hash_str_find(Z_ARR_P(config_zv), "grid", sizeof("grid") - 1);
        zval *block_zv = zend_hash_str_find(Z_ARR_P(config_zv), "block", sizeof("block") - 1);

        if (grid_zv && Z_TYPE_P(grid_zv) == IS_ARRAY)
        {
            zval *x = zend_hash_index_find(Z_ARR_P(grid_zv), 0);
            zval *y = zend_hash_index_find(Z_ARR_P(grid_zv), 1);
            zval *z = zend_hash_index_find(Z_ARR_P(grid_zv), 2);
            if (x)
                grid[0] = zval_get_long(x);
            if (y)
                grid[1] = zval_get_long(y);
            if (z)
                grid[2] = zval_get_long(z);
        }

        if (block_zv && Z_TYPE_P(block_zv) == IS_ARRAY)
        {
            zval *x = zend_hash_index_find(Z_ARR_P(block_zv), 0);
            zval *y = zend_hash_index_find(Z_ARR_P(block_zv), 1);
            zval *z = zend_hash_index_find(Z_ARR_P(block_zv), 2);
            if (x)
                block[0] = zval_get_long(x);
            if (y)
                block[1] = zval_get_long(y);
            if (z)
                block[2] = zval_get_long(z);
        }
    }
}

static zend_bool module_should_use_async(int grid[3], int block[3])
{
    int total_threads = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2];
    return total_threads >= MIN_KERNEL_SIZE_FOR_ASYNC;
}

static void module_cleanup_args_and_buffers(zval *args, void **cuda_args,
                                            void **temp_buffers, int temp_buffers_count)
{
    if (temp_buffers)
    {
        for (int i = 0; i < temp_buffers_count; i++)
        {
            if (temp_buffers[i])
            {
                efree(temp_buffers[i]);
            }
        }
        efree(temp_buffers);
    }

    if (cuda_args)
        efree(cuda_args);

    if (args)
        efree(args);
}

static zend_bool module_initialize_global_cuda(cuda_module_object *module)
{
    pthread_mutex_lock(&g_cuda_global_init_mutex);

    if (g_cuda_initialized)
    {
        pthread_mutex_unlock(&g_cuda_global_init_mutex);
        return 1;
    }

    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS)
    {
        pthread_mutex_unlock(&g_cuda_global_init_mutex);
        module_log_error("Failed to initialize CUDA driver: %s",
                         module_get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuDeviceGet(&g_primary_device, 0);
    if (cu_result != CUDA_SUCCESS)
    {
        pthread_mutex_unlock(&g_cuda_global_init_mutex);
        module_log_error("Failed to get CUDA device: %s",
                         module_get_cuda_error_string(cu_result));
        return 0;
    }

    g_cuda_initialized = 1;
    pthread_mutex_unlock(&g_cuda_global_init_mutex);

    return 1;
}

static void module_initialize_stream_pool(cuda_module_object *module)
{
    if (!module->stream_pool)
    {
        module->stream_pool = (stream_pool_t *)emalloc(sizeof(stream_pool_t));
        memset(module->stream_pool, 0, sizeof(stream_pool_t));
        pthread_mutex_init(&module->stream_pool->mutex, NULL);
    }

    if (module->stream_pool->streams)
        return;

    module->stream_pool->actives = 0;
    module->stream_pool->capacity = INITIAL_STREAM_POOL_SIZE;
    module->stream_pool->size = INITIAL_STREAM_POOL_SIZE;
    module->stream_pool->streams = (pooled_stream_t *)ecalloc(module->stream_pool->capacity, sizeof(pooled_stream_t));

    for (int i = 0; i < INITIAL_STREAM_POOL_SIZE; i++)
    {
        CUresult cu_result = cuStreamCreate(&module->stream_pool->streams[i].stream, CU_STREAM_NON_BLOCKING);
        if (cu_result != CUDA_SUCCESS)
        {
            module_log_error("Failed to create stream %d for pool: %s",
                             i, module_get_cuda_error_string(cu_result));
            module->stream_pool->streams[i].stream = NULL;
        }
        else
        {
            module->stream_pool->streams[i].in_use = 0;
            module->stream_pool->streams[i].last_used = module_get_current_time_ms();
        }
    }
}

static void module_destroy_stream_pool(cuda_module_object *module)
{
    if (!module->stream_pool)
        return;

    if (module->stream_pool->streams)
    {
        pthread_mutex_lock(&module->stream_pool->mutex);

        for (int i = 0; i < module->stream_pool->size; i++)
        {
            if (module->stream_pool->streams[i].stream)
            {
                CUresult result = cuStreamDestroy(module->stream_pool->streams[i].stream);
                if (result != CUDA_SUCCESS && result != CUDA_ERROR_INVALID_HANDLE)
                {
                    module_log_error("Failed to destroy stream from pool: %s",
                                     module_get_cuda_error_string(result));
                }
            }
        }

        module->stream_pool->actives = 0;
        efree(module->stream_pool->streams);
        module->stream_pool->streams = NULL;
        module->stream_pool->size = 0;
        module->stream_pool->capacity = 0;

        pthread_mutex_unlock(&module->stream_pool->mutex);
    }

    pthread_mutex_destroy(&module->stream_pool->mutex);
    efree(module->stream_pool);
    module->stream_pool = NULL;
}

static zend_bool module_validate_async_operation_count(cuda_module_object *module)
{
    if (module->stream_pool->actives >= MAX_STREAM_POOL_SIZE)
    {
        return 0;
    }

    return 1;
}

static CUstream module_get_stream_from_pool(cuda_module_object *module)
{
    if (!module->stream_pool)
        return NULL;

    CUstream stream = NULL;
    double current_time = module_get_current_time_ms();

    pthread_mutex_lock(&module->stream_pool->mutex);

    for (int i = 0; i < module->stream_pool->size; i++)
    {
        if (!module->stream_pool->streams[i].in_use &&
            module->stream_pool->streams[i].stream != NULL)
        {
            module->stream_pool->streams[i].in_use = 1;
            module->stream_pool->streams[i].last_used = current_time;
            stream = module->stream_pool->streams[i].stream;
            break;
        }
    }

    if (!stream && module->stream_pool->size < MAX_STREAM_POOL_SIZE)
    {
        if (module->stream_pool->size >= module->stream_pool->capacity)
        {
            int new_capacity = module->stream_pool->capacity * 2;
            if (new_capacity > MAX_STREAM_POOL_SIZE)
                new_capacity = MAX_STREAM_POOL_SIZE;

            pooled_stream_t *new_streams = (pooled_stream_t *)erealloc(
                module->stream_pool->streams,
                new_capacity * sizeof(pooled_stream_t));

            if (new_streams)
            {
                module->stream_pool->streams = new_streams;
                module->stream_pool->capacity = new_capacity;
            }
        }

        if (module->stream_pool->size < module->stream_pool->capacity)
        {
            CUresult cu_result = cuStreamCreate(
                &module->stream_pool->streams[module->stream_pool->size].stream,
                CU_STREAM_NON_BLOCKING);

            if (cu_result == CUDA_SUCCESS)
            {
                module->stream_pool->streams[module->stream_pool->size].in_use = 1;
                module->stream_pool->streams[module->stream_pool->size].last_used = current_time;
                stream = module->stream_pool->streams[module->stream_pool->size].stream;
                module->stream_pool->size++;
            }
        }
    }

    pthread_mutex_unlock(&module->stream_pool->mutex);

    if (!stream)
    {
        CUresult cu_result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
        if (cu_result != CUDA_SUCCESS)
        {
            module_log_error("Failed to create stream: %s",
                             module_get_cuda_error_string(cu_result));
            return NULL;
        }
    }

    module->stream_pool->actives++;
    return stream;
}

static void module_return_stream_to_pool(cuda_module_object *module, CUstream stream)
{
    if (!stream || !module->stream_pool)
        return;

    pthread_mutex_lock(&module->stream_pool->mutex);

    for (int i = 0; i < module->stream_pool->size; i++)
    {
        if (module->stream_pool->streams[i].stream == stream)
        {
            module->stream_pool->actives--;
            module->stream_pool->streams[i].in_use = 0;
            module->stream_pool->streams[i].last_used = module_get_current_time_ms();
            break;
        }
    }

    pthread_mutex_unlock(&module->stream_pool->mutex);
}

static zend_bool module_initialize_cuda_context(cuda_module_object *module)
{
    CUresult cu_result;

    if (!module_initialize_global_cuda(module))
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to initialize CUDA driver");
        return 0;
    }

    if (module->cu_context)
    {
        CUcontext current;
        cu_result = cuCtxGetCurrent(&current);
        if (cu_result != CUDA_SUCCESS || current != module->cu_context)
        {
            cu_result = cuCtxSetCurrent(module->cu_context);
            if (cu_result != CUDA_SUCCESS)
            {
                module_check_cuda_error(module, cu_result, "setting current context");
                return 0;
            }
        }
        return 1;
    }

    unsigned int ctx_flags = CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST;

    cu_result = cuCtxCreate(&module->cu_context, ctx_flags, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to create CUDA context: %s",
                                module_get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuStreamCreate(&module->cu_stream, CU_STREAM_DEFAULT);
    if (cu_result != CUDA_SUCCESS)
    {
        module_check_cuda_error(module, cu_result, "creating default stream");
        cuCtxDestroy(module->cu_context);
        module->cu_context = NULL;
        return 0;
    }

    module_initialize_stream_pool(module);

    return 1;
}

static void module_cleanup_cuda_resources(cuda_module_object *module)
{
    CUresult cu_result;

    if (module->loaded_modules)
    {
        zend_string *key;
        CUmodule *cu_module_ptr;

        ZEND_HASH_FOREACH_STR_KEY_PTR(module->loaded_modules, key, cu_module_ptr)
        {
            if (cu_module_ptr && *cu_module_ptr)
            {
                cu_result = cuModuleUnload(*cu_module_ptr);
                if (cu_result != CUDA_SUCCESS && cu_result != CUDA_ERROR_INVALID_HANDLE)
                {
                    module_log_error("Failed to unload CUDA module: %s",
                                     module_get_cuda_error_string(cu_result));
                }
                efree(cu_module_ptr);
            }
        }
        ZEND_HASH_FOREACH_END();

        zend_hash_destroy(module->loaded_modules);
        efree(module->loaded_modules);
        module->loaded_modules = NULL;
    }

    if (module->cu_stream)
    {
        cu_result = cuStreamDestroy(module->cu_stream);
        if (cu_result != CUDA_SUCCESS)
        {
            module_log_error("Failed to destroy CUDA stream: %s",
                             module_get_cuda_error_string(cu_result));
        }
        module->cu_stream = NULL;
    }

    if (module->cu_context)
    {
        cu_result = cuCtxDestroy(module->cu_context);
        if (cu_result != CUDA_SUCCESS)
        {
            module_log_error("Failed to destroy CUDA context: %s",
                             module_get_cuda_error_string(cu_result));
        }
        module->cu_context = NULL;
    }
}

static CUmodule module_get_or_load_module(cuda_module_object *module, zend_string *kernel_name)
{
    if (!module->ptx_code || module->ptx_size == 0)
    {
        zend_throw_exception_ex(NULL, 0,
                                "No PTX code available for kernel '%s'",
                                ZSTR_VAL(kernel_name));
        return NULL;
    }

    CUmodule *cached_module = (CUmodule *)zend_hash_find_ptr(module->loaded_modules, kernel_name);
    if (cached_module)
    {
        return *cached_module;
    }

    CUresult cu_result;
    CUmodule cu_module = NULL;

    cu_result = cuModuleLoadDataEx(&cu_module, module->ptx_code, 0, NULL, NULL);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to load PTX module for kernel '%s': %s",
                                ZSTR_VAL(kernel_name),
                                module_get_cuda_error_string(cu_result));
        return NULL;
    }

    CUmodule *module_ptr = (CUmodule *)emalloc(sizeof(CUmodule));
    *module_ptr = cu_module;

    if (zend_hash_add_ptr(module->loaded_modules, kernel_name, module_ptr) == NULL)
    {
        efree(module_ptr);
        cuModuleUnload(cu_module);
        zend_throw_exception_ex(NULL, 0,
                                "Failed to cache module for kernel '%s'",
                                ZSTR_VAL(kernel_name));
        return NULL;
    }

    return cu_module;
}

static zend_bool module_validate_launch_config(cuda_module_object *module, int grid[3], int block[3])
{
    int max_threads, max_block_dims[3], max_grid_dims[3];
    CUresult cu_result;

    cu_result = cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_block_dims[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_block_dims[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_block_dims[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_grid_dims[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_grid_dims[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_grid_dims[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    if (block[0] <= 0 || block[1] <= 0 || block[2] <= 0)
    {
        return 0;
    }

    if (block[0] > max_block_dims[0] || block[1] > max_block_dims[1] || block[2] > max_block_dims[2])
    {
        return 0;
    }

    int total_threads_per_block = block[0] * block[1] * block[2];
    if (total_threads_per_block > max_threads)
    {
        return 0;
    }

    if (grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0)
    {
        return 0;
    }

    if (grid[0] > max_grid_dims[0] || grid[1] > max_grid_dims[1] || grid[2] > max_grid_dims[2])
    {
        return 0;
    }

    return 1;
}

static void module_cleanup_timeout_operations(cuda_module_object *module)
{
    static double last_check = 0;
    double current_time = module_get_current_time_ms();

    if (current_time - last_check < ERROR_CHECK_INTERVAL)
        return;

    last_check = current_time;

    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && op->is_active)
        {
            CUresult result = cuStreamQuery(op->stream);
            if (result == CUDA_SUCCESS)
            {
                module_cleanup_async_operation_by_id(module, op->id);
                op->is_active = 0;
                continue;
            }

            double elapsed = current_time - op->start_time;
            if (elapsed > ASYNC_OP_TIMEOUT_MS)
            {
                CUresult sync_result = cuStreamSynchronize(op->stream);
                if (sync_result != CUDA_SUCCESS)
                {
                    module_log_error("Failed to synchronize stream after timeout: %s",
                                     module_get_cuda_error_string(sync_result));
                }
                else
                {
                    op->is_active = 0;
                }

                module_cleanup_async_operation_by_id(module, op->id);
            }
        }
    }
    ZEND_HASH_FOREACH_END();
}

static int module_create_async_operation(cuda_module_object *module,
                                         zend_string *kernel_name,
                                         void **cuda_args,
                                         void **temp_buffers,
                                         int temp_buffers_count,
                                         int grid[3],
                                         int block[3],
                                         int argc)
{
    if (!module_initialize_cuda_context(module))
    {
        return 0;
    }

    cuda_async_operation *op = (cuda_async_operation *)emalloc(sizeof(cuda_async_operation));
    if (!op)
        return 0;

    memset(op, 0, sizeof(cuda_async_operation));

    op->id = module->next_async_op_id++;
    op->kernel_name = kernel_name ? zend_string_copy(kernel_name) : NULL;
    op->cuda_args = cuda_args;
    op->temp_buffers = temp_buffers;
    op->temp_buffers_count = temp_buffers_count;
    op->is_active = 1;
    op->start_time = module_get_current_time_ms();
    op->argc = argc;

    memcpy(op->grid, grid, sizeof(int) * 3);
    memcpy(op->block, block, sizeof(int) * 3);

    op->stream = module_get_stream_from_pool(module);
    if (!op->stream)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to get CUDA stream for async operation");
        if (op->kernel_name)
            zend_string_release(op->kernel_name);
        efree(op);
        return 0;
    }

    op->last_error = CUDA_SUCCESS;
    memset(op->error_message, 0, sizeof(op->error_message));

    zend_hash_index_update_ptr(module->async_operations, op->id, op);

    return op->id;
}

static void module_cleanup_async_operation_by_id(cuda_module_object *module, int op_id)
{
    cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
    if (!op)
    {
        return;
    }

    if (op->stream)
    {
        module_return_stream_to_pool(module, op->stream);
    }

    if (op->temp_buffers)
    {
        for (int i = 0; i < op->temp_buffers_count; i++)
        {
            if (op->temp_buffers[i])
            {
                efree(op->temp_buffers[i]);
            }
        }
        efree(op->temp_buffers);
    }

    if (op->cuda_args)
    {
        efree(op->cuda_args);
    }

    if (op->kernel_name)
    {
        zend_string_release(op->kernel_name);
    }

    zend_hash_index_del(module->async_operations, op_id);
    efree(op);
}

static void module_cleanup_all_async_operations(cuda_module_object *module)
{
    if (!module->async_operations)
        return;

    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op)
        {
            module_cleanup_async_operation_by_id(module, op->id);
        }
    }
    ZEND_HASH_FOREACH_END();
}

static zend_bool module_prepare_cuda_arguments(cuda_kernel_data *kernel, zval *args, int argc,
                                               void ***cuda_args_ptr, void ***temp_buffers_ptr,
                                               int *temp_buffers_count_ptr)
{
    if (!kernel || !kernel->parameters || argc != kernel->parameters->total)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Argument count mismatch: expected %d, got %d",
                                kernel->parameters ? kernel->parameters->total : 0,
                                argc);
        return 0;
    }

    void **cuda_args = (void **)emalloc(sizeof(void *) * argc);
    void **temp_gpu_buffers = (void **)emalloc(sizeof(void *) * argc);
    int temp_buffers_count = 0;
    zend_bool valid = 1;

    for (int i = 0; i < argc && valid; i++)
    {
        func_parameter *param = kernel->parameters->parameters[i];
        zval *arg = &args[i];

        switch (param->type)
        {
        case INPUT:
        case PARAMETER:
            if (param->dtype == LIST)
            {
                if (Z_TYPE_P(arg) != IS_OBJECT ||
                    !instanceof_function(Z_OBJCE_P(arg), cuda_array_ce))
                {
                    zend_throw_exception_ex(NULL, 0,
                                            "Argument %d '%s' must be a CudaArray",
                                            i + 1, param->name);
                    valid = 0;
                    break;
                }

                cuda_array_obj *array_obj = (cuda_array_obj *)((char *)Z_OBJ_P(arg) - XtOffsetOf(cuda_array_obj, obj));
                if (!array_obj->tensor_handle)
                {
                    zend_throw_exception_ex(NULL, 0,
                                            "Argument %d '%s': CudaArray has no tensor data",
                                            i + 1, param->name);
                    valid = 0;
                    break;
                }

                tensor_t *tensor = array_obj->tensor_handle;
                if (tensor->dtype != param->second_dtype)
                {
                    const char *expected = module_dtype_to_string(param->second_dtype);
                    const char *actual = module_dtype_to_string(tensor->dtype);
                    zend_throw_exception_ex(NULL, 0,
                                            "Argument %d '%s': expected dtype %s, got %s",
                                            i + 1, param->name, expected, actual);
                    valid = 0;
                    break;
                }

                cuda_args[i] = &tensor->data;
            }
            else
            {
                switch (param->dtype)
                {
                case INT32:
                {
                    int *int_ptr = (int *)emalloc(sizeof(int));
                    *int_ptr = (int)zval_get_long(arg);
                    cuda_args[i] = int_ptr;
                    temp_gpu_buffers[temp_buffers_count++] = int_ptr;
                    break;
                }
                case FLOAT32:
                {
                    float *float_ptr = (float *)emalloc(sizeof(float));
                    *float_ptr = (float)zval_get_double(arg);
                    cuda_args[i] = float_ptr;
                    temp_gpu_buffers[temp_buffers_count++] = float_ptr;
                    break;
                }
                case FLOAT64:
                {
                    double *double_ptr = (double *)emalloc(sizeof(double));
                    *double_ptr = zval_get_double(arg);
                    cuda_args[i] = double_ptr;
                    temp_gpu_buffers[temp_buffers_count++] = double_ptr;
                    break;
                }
                case INT64:
                {
                    zend_long *long_ptr = (zend_long *)emalloc(sizeof(zend_long));
                    *long_ptr = zval_get_long(arg);
                    cuda_args[i] = long_ptr;
                    temp_gpu_buffers[temp_buffers_count++] = long_ptr;
                    break;
                }
                case BOOL:
                {
                    int *bool_ptr = (int *)emalloc(sizeof(int));
                    *bool_ptr = zval_is_true(arg) ? 1 : 0;
                    cuda_args[i] = bool_ptr;
                    temp_gpu_buffers[temp_buffers_count++] = bool_ptr;
                    break;
                }
                default:
                    zend_throw_exception_ex(NULL, 0,
                                            "Unsupported scalar dtype for argument %d",
                                            i + 1);
                    valid = 0;
                    break;
                }
            }
            break;

        case OUTPUT:
            if (Z_TYPE_P(arg) != IS_OBJECT ||
                !instanceof_function(Z_OBJCE_P(arg), cuda_array_ce))
            {
                zend_throw_exception_ex(NULL, 0,
                                        "Output argument %d '%s' must be a CudaArray",
                                        i + 1, param->name);
                valid = 0;
                break;
            }

            cuda_array_obj *array_obj = (cuda_array_obj *)((char *)Z_OBJ_P(arg) - XtOffsetOf(cuda_array_obj, obj));
            if (!array_obj->tensor_handle)
            {
                zend_throw_exception_ex(NULL, 0,
                                        "Output argument %d '%s': CudaArray has no tensor data",
                                        i + 1, param->name);
                valid = 0;
                break;
            }

            tensor_t *tensor = array_obj->tensor_handle;
            if (tensor->dtype != param->second_dtype)
            {
                const char *expected = module_dtype_to_string(param->second_dtype);
                const char *actual = module_dtype_to_string(tensor->dtype);
                zend_throw_exception_ex(NULL, 0,
                                        "Output argument %d '%s': expected dtype %s, got %s",
                                        i + 1, param->name, expected, actual);
                valid = 0;
                break;
            }

            cuda_args[i] = &tensor->data;
            tensor->is_dirty = 1;
            break;

        default:
            zend_throw_exception_ex(NULL, 0,
                                    "Unknown parameter type for argument %d",
                                    i + 1);
            valid = 0;
            break;
        }
    }

    if (valid)
    {
        *cuda_args_ptr = cuda_args;
        *temp_buffers_ptr = temp_gpu_buffers;
        *temp_buffers_count_ptr = temp_buffers_count;
    }
    else
    {
        module_cleanup_args_and_buffers(NULL, cuda_args, temp_gpu_buffers, temp_buffers_count);
    }

    return valid;
}

static zend_bool module_execute_cuda_kernel(cuda_module_object *module,
                                            cuda_kernel_data *kernel,
                                            int grid[3], int block[3],
                                            void **cuda_args, int argc,
                                            CUstream stream)
{
    CUresult cu_result;
    CUfunction cu_function;

    if (!module_initialize_cuda_context(module))
    {
        return 0;
    }

    if (!module_validate_launch_config(module, grid, block))
    {
        zend_throw_exception_ex(NULL, 0, "Invalid grid/block configuration");
        return 0;
    }

    CUmodule cu_module = module_get_or_load_module(module, kernel->name);
    if (!cu_module)
    {
        return 0;
    }

    cu_result = cuModuleGetFunction(&cu_function, cu_module, ZSTR_VAL(kernel->name));
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to get kernel function '%s': %s",
                                ZSTR_VAL(kernel->name),
                                module_get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuLaunchKernel(cu_function,
                               grid[0], grid[1], grid[2],
                               block[0], block[1], block[2],
                               0,
                               stream,
                               cuda_args,
                               NULL);

    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to launch kernel '%s': %s",
                                ZSTR_VAL(kernel->name),
                                module_get_cuda_error_string(cu_result));
        return 0;
    }

    return 1;
}

ZEND_METHOD(CompiledModule, initialize)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (!module->ptx_code || module->ptx_size == 0)
    {
        zend_throw_exception_ex(NULL, 0, "No PTX code available");
        RETURN_FALSE;
    }

    if (!module_initialize_cuda_context(module))
    {
        RETURN_FALSE;
    }

    if (module->kernel_functions)
    {
        zend_string *key;
        cuda_kernel_data *kernel_data;

        ZEND_HASH_FOREACH_STR_KEY_PTR(module->kernel_functions, key, kernel_data)
        {
            if (kernel_data)
            {
                CUmodule cu_module = module_get_or_load_module(module, kernel_data->name);
                if (!cu_module)
                {
                    zend_throw_exception_ex(NULL, 0,
                                            "Failed to pre-load kernel '%s'",
                                            ZSTR_VAL(kernel_data->name));
                    RETURN_FALSE;
                }
            }
        }
        ZEND_HASH_FOREACH_END();
    }

    RETURN_TRUE;
}

ZEND_METHOD(CompiledModule, run)
{
    zend_string *kernel_name;
    zval *config_zv = NULL;
    zval *args_zv = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 3)
    Z_PARAM_STR(kernel_name)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY_OR_NULL(config_zv)
    Z_PARAM_ARRAY_OR_NULL(args_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (!module->ptx_code || module->ptx_size == 0)
    {
        zend_throw_exception_ex(NULL, 0, "No PTX code available");
        RETURN_FALSE;
    }

    cuda_kernel_data *kernel = zend_hash_find_ptr(module->kernel_functions, kernel_name);
    if (!kernel)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel '%s' not found in compiled module",
                                ZSTR_VAL(kernel_name));
        RETURN_FALSE;
    }

    int argc = 0;
    zval *args = NULL;

    if (args_zv && Z_TYPE_P(args_zv) == IS_ARRAY)
    {
        HashTable *ht = Z_ARRVAL_P(args_zv);
        argc = zend_hash_num_elements(ht);

        if (argc > 0)
        {
            args = (zval *)emalloc(sizeof(zval) * argc);
            int i = 0;
            zval *val;

            ZEND_HASH_FOREACH_VAL(ht, val)
            {
                ZVAL_COPY(&args[i++], val);
            }
            ZEND_HASH_FOREACH_END();
        }
    }

    int expected_args = kernel->parameters ? kernel->parameters->total : 0;
    if (expected_args != argc)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel '%s' expects %d arguments, %d given",
                                ZSTR_VAL(kernel_name),
                                expected_args, argc);
        if (args)
            efree(args);
        RETURN_FALSE;
    }

    int grid[3], block[3];
    module_prepare_launch_config(config_zv, grid, block);

    void **cuda_args = NULL;
    void **temp_gpu_buffers = NULL;
    int temp_buffers_count = 0;

    zend_bool args_prepared = module_prepare_cuda_arguments(kernel, args, argc,
                                                            &cuda_args, &temp_gpu_buffers,
                                                            &temp_buffers_count);
    if (!args_prepared)
    {
        if (args)
            efree(args);
        RETURN_FALSE;
    }

    double start_time = module_get_current_time_ms();
    zend_bool success = module_execute_cuda_kernel(module, kernel, grid, block,
                                                   cuda_args, argc, module->cu_stream);

    if (success)
    {
        CUresult cu_result = cuStreamSynchronize(module->cu_stream);
        if (cu_result != CUDA_SUCCESS)
        {
            module_check_cuda_error(module, cu_result, "kernel execution");
            success = 0;
        }
        else
        {
            module->kernel_execution_count++;
            module->total_execution_time_ms += (module_get_current_time_ms() - start_time);
        }
    }

    module_cleanup_args_and_buffers(args, cuda_args, temp_gpu_buffers, temp_buffers_count);
    RETURN_BOOL(success);
}

ZEND_METHOD(CompiledModule, runAsync)
{
    zend_string *kernel_name;
    zval *config_zv = NULL;
    zval *args_zv = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 3)
    Z_PARAM_STR(kernel_name)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY_OR_NULL(config_zv)
    Z_PARAM_ARRAY_OR_NULL(args_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    if (!module->ptx_code || module->ptx_size == 0)
    {
        zend_throw_exception_ex(NULL, 0, "No PTX code available");
        RETURN_FALSE;
    }

    if (!module_initialize_cuda_context(module))
    {
        RETURN_FALSE;
    }

    if (!module_validate_async_operation_count(module))
    {
        zend_throw_exception_ex(NULL, 0,
                                "Too many active async operations (%d maximum)",
                                MAX_STREAM_POOL_SIZE);
        RETURN_FALSE;
    }

    cuda_kernel_data *kernel = zend_hash_find_ptr(module->kernel_functions, kernel_name);
    if (!kernel)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel '%s' not found in compiled module",
                                ZSTR_VAL(kernel_name));
        RETURN_FALSE;
    }

    int argc = 0;
    zval *args = NULL;

    if (args_zv && Z_TYPE_P(args_zv) == IS_ARRAY)
    {
        HashTable *ht = Z_ARRVAL_P(args_zv);
        argc = zend_hash_num_elements(ht);

        if (argc > 0)
        {
            args = (zval *)emalloc(sizeof(zval) * argc);
            int i = 0;
            zval *val;

            ZEND_HASH_FOREACH_VAL(ht, val)
            {
                ZVAL_COPY(&args[i++], val);
            }
            ZEND_HASH_FOREACH_END();
        }
    }

    int expected_args = kernel->parameters ? kernel->parameters->total : 0;
    if (expected_args != argc)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel '%s' expects %d arguments, %d given",
                                ZSTR_VAL(kernel_name),
                                expected_args, argc);
        if (args)
            efree(args);
        RETURN_FALSE;
    }

    int grid[3], block[3];
    module_prepare_launch_config(config_zv, grid, block);

    if (!module_should_use_async(grid, block))
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel too small for async execution. Use run() instead.");
        if (args)
            efree(args);
        RETURN_FALSE;
    }

    void **cuda_args = NULL;
    void **temp_gpu_buffers = NULL;
    int temp_buffers_count = 0;

    zend_bool args_prepared = module_prepare_cuda_arguments(kernel, args, argc,
                                                            &cuda_args, &temp_gpu_buffers,
                                                            &temp_buffers_count);
    if (!args_prepared)
    {
        if (args)
            efree(args);
        RETURN_FALSE;
    }

    int op_id = module_create_async_operation(module, kernel_name,
                                              cuda_args, temp_gpu_buffers,
                                              temp_buffers_count, grid, block, argc);
    if (!op_id)
    {
        module_cleanup_args_and_buffers(args, cuda_args, temp_gpu_buffers, temp_buffers_count);
        RETURN_FALSE;
    }

    cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
    if (!op)
    {
        module_cleanup_args_and_buffers(args, cuda_args, temp_gpu_buffers, temp_buffers_count);
        RETURN_FALSE;
    }

    double start_time = module_get_current_time_ms();
    zend_bool success = module_execute_cuda_kernel(module, kernel, grid, block,
                                                   cuda_args, argc, op->stream);

    if (args)
        efree(args);

    if (success)
    {
        module->kernel_execution_count++;
        module->total_execution_time_ms += (module_get_current_time_ms() - start_time);
        module->has_pending_operations = 1;
        RETURN_LONG(op_id);
    }
    else
    {
        module_cleanup_async_operation_by_id(module, op_id);
        RETURN_FALSE;
    }
}

ZEND_METHOD(CompiledModule, sync)
{
    zend_long op_id = -1;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    CUresult cu_result;

    module_cleanup_timeout_operations(module);

    if (op_id == -1)
    {
        zend_ulong num_idx;
        cuda_async_operation *op;
        zend_bool all_success = 1;

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op && op->is_active)
            {
                cu_result = cuStreamSynchronize(op->stream);
                if (cu_result != CUDA_SUCCESS)
                {
                    module_check_cuda_error(module, cu_result, "stream synchronization");
                    all_success = 0;
                }

                module_cleanup_async_operation_by_id(module, op->id);
                op->is_active = 0;
            }
        }
        ZEND_HASH_FOREACH_END();

        module->has_pending_operations = 0;
        RETURN_BOOL(all_success);
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
        if (!op)
        {
            zend_throw_exception_ex(NULL, 0, "Async operation %ld not found", op_id);
            RETURN_FALSE;
        }

        if (op->is_active)
        {
            cu_result = cuStreamSynchronize(op->stream);
            if (cu_result != CUDA_SUCCESS)
            {
                module_check_cuda_error(module, cu_result, "stream synchronization");
                RETURN_FALSE;
            }
            op->is_active = 0;
        }

        zend_ulong num_idx;
        cuda_async_operation *check_op;
        module->has_pending_operations = 0;
        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, check_op)
        {
            if (check_op && check_op->is_active)
            {
                module->has_pending_operations = 1;
                break;
            }
        }
        ZEND_HASH_FOREACH_END();

        RETURN_TRUE;
    }
}

ZEND_METHOD(CompiledModule, isFinished)
{
    zend_long op_id = -1;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    module_cleanup_timeout_operations(module);

    if (op_id == -1)
    {
        zend_ulong num_idx;
        cuda_async_operation *op;
        zend_bool all_finished = 1;

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op && op->is_active)
            {
                CUresult cu_result = cuStreamQuery(op->stream);

                if (cu_result == CUDA_SUCCESS)
                {
                    op->is_active = 0;
                }
                else if (cu_result == CUDA_ERROR_NOT_READY)
                {
                    all_finished = 0;
                }
                else
                {
                    module_check_cuda_error(module, cu_result, "stream query");
                    op->is_active = 0;
                }
            }
        }
        ZEND_HASH_FOREACH_END();

        module->has_pending_operations = !all_finished;
        RETURN_BOOL(all_finished);
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
        if (!op)
        {
            RETURN_TRUE;
        }

        if (!op->is_active)
        {
            RETURN_TRUE;
        }

        CUresult cu_result = cuStreamQuery(op->stream);

        if (cu_result == CUDA_SUCCESS)
        {
            op->is_active = 0;
            RETURN_TRUE;
        }
        else if (cu_result == CUDA_ERROR_NOT_READY)
        {
            RETURN_FALSE;
        }
        else
        {
            module_check_cuda_error(module, cu_result, "stream query");
            op->is_active = 0;
            RETURN_FALSE;
        }
    }
}

ZEND_METHOD(CompiledModule, wait)
{
    zend_long op_id = -1;
    zend_long timeout_ms = -1;

    ZEND_PARSE_PARAMETERS_START(0, 2)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    Z_PARAM_LONG(timeout_ms)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    module_cleanup_timeout_operations(module);

    double start_time = module_get_current_time_ms();

    if (op_id == -1)
    {
        zend_bool all_completed = 1;

        while (1)
        {
            all_completed = 1;
            zend_ulong num_idx;
            cuda_async_operation *op;

            ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
            {
                if (op && op->is_active)
                {
                    CUresult cu_result = cuStreamQuery(op->stream);

                    if (cu_result == CUDA_SUCCESS)
                    {
                        module_cleanup_async_operation_by_id(module, op->id);
                        op->is_active = 0;
                    }
                    else if (cu_result == CUDA_ERROR_NOT_READY)
                    {
                        all_completed = 0;
                    }
                    else
                    {
                        module_check_cuda_error(module, cu_result, "stream query");
                        op->is_active = 0;
                    }
                }
            }
            ZEND_HASH_FOREACH_END();

            if (all_completed)
            {
                module->has_pending_operations = 0;
                RETURN_TRUE;
            }

            if (timeout_ms >= 0)
            {
                double elapsed = module_get_current_time_ms() - start_time;
                if (elapsed > timeout_ms)
                {
                    zend_throw_exception_ex(NULL, 0,
                                            "Timeout waiting for all async operations after %.2f ms",
                                            elapsed);
                    RETURN_FALSE;
                }
            }

            usleep(1000);
        }
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
        if (!op)
        {
            RETURN_TRUE;
        }

        if (!op->is_active)
        {
            RETURN_TRUE;
        }

        while (op->is_active)
        {
            CUresult cu_result = cuStreamQuery(op->stream);

            if (cu_result == CUDA_SUCCESS)
            {
                op->is_active = 0;
                RETURN_TRUE;
            }
            else if (cu_result != CUDA_ERROR_NOT_READY)
            {
                module_check_cuda_error(module, cu_result, "stream query");
                op->is_active = 0;
                RETURN_FALSE;
            }

            if (timeout_ms >= 0)
            {
                double elapsed = module_get_current_time_ms() - start_time;
                if (elapsed > timeout_ms)
                {
                    op->is_active = 0;
                    zend_throw_exception_ex(NULL, 0,
                                            "Timeout waiting for async operation %ld after %.2f ms",
                                            op_id, elapsed);
                    RETURN_FALSE;
                }
            }

            usleep(1000);
        }

        RETURN_TRUE;
    }
}

ZEND_METHOD(CompiledModule, hasKernel)
{
    zend_string *name;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    RETURN_BOOL(zend_hash_exists(module->kernel_functions, name));
}

ZEND_METHOD(CompiledModule, getKernels)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    array_init(return_value);

    zend_string *key;
    ZEND_HASH_FOREACH_STR_KEY(module->kernel_functions, key)
    {
        add_next_index_string(return_value, ZSTR_VAL(key));
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(CompiledModule, getPtx)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (module->ptx_code)
    {
        RETURN_STRING(module->ptx_code);
    }
    else
    {
        RETURN_NULL();
    }
}

ZEND_METHOD(CompiledModule, save)
{
    zend_string *filename;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(filename)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (!module->ptx_code)
    {
        RETURN_FALSE;
    }

    FILE *file = fopen(ZSTR_VAL(filename), "w");
    if (!file)
    {
        php_error_docref(NULL, E_WARNING, "Failed to open file for writing: %s",
                         ZSTR_VAL(filename));
        RETURN_FALSE;
    }

    size_t written = fwrite(module->ptx_code, 1, module->ptx_size, file);
    fclose(file);

    RETURN_BOOL(written == module->ptx_size);
}

ZEND_METHOD(CompiledModule, __serialize)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    array_init(return_value);

    if (module->ptx_code && module->ptx_size > 0)
    {
        zend_string *b64_zstr = php_base64_encode(
            (unsigned char *)module->ptx_code,
            module->ptx_size);

        if (!b64_zstr)
        {
            zend_throw_exception_ex(NULL, 0, "Failed to encode PTX to base64");
            return;
        }

        add_assoc_str(return_value, "ptx_b64", b64_zstr);
        add_assoc_long(return_value, "ptx_size", module->ptx_size);
    }
    else
    {
        add_assoc_null(return_value, "ptx_b64");
    }

    zval kernels_zv;
    array_init(&kernels_zv);

    zend_string *key;
    cuda_kernel_data *kernel_data;

    ZEND_HASH_FOREACH_STR_KEY_PTR(module->kernel_functions, key, kernel_data)
    {
        if (kernel_data)
        {
            zval kernel_zv;
            array_init(&kernel_zv);

            add_assoc_string(&kernel_zv, "name", ZSTR_VAL(kernel_data->name));

            if (kernel_data->parameters)
            {
                zval params_zv;
                array_init(&params_zv);

                for (int i = 0; i < kernel_data->parameters->total; i++)
                {
                    func_parameter *param = kernel_data->parameters->parameters[i];
                    zval param_zv;
                    array_init(&param_zv);

                    add_assoc_string(&param_zv, "name", param->name);
                    add_assoc_long(&param_zv, "type", param->type);
                    add_assoc_long(&param_zv, "dtype", param->dtype);
                    add_assoc_long(&param_zv, "second_dtype", param->second_dtype);

                    add_next_index_zval(&params_zv, &param_zv);
                }

                add_assoc_zval(&kernel_zv, "parameters", &params_zv);
            }

            add_assoc_zval(&kernels_zv, ZSTR_VAL(key), &kernel_zv);
        }
    }
    ZEND_HASH_FOREACH_END();

    add_assoc_zval(return_value, "kernels", &kernels_zv);
}

ZEND_METHOD(CompiledModule, __unserialize)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    zval *data;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(data)
    ZEND_PARSE_PARAMETERS_END();

    if (module->ptx_code)
    {
        efree(module->ptx_code);
        module->ptx_code = NULL;
    }

    if (module->kernel_functions)
    {
        zend_string *key;
        cuda_kernel_data *kernel_data;

        ZEND_HASH_FOREACH_STR_KEY_PTR(module->kernel_functions, key, kernel_data)
        {
            if (kernel_data)
            {
                if (kernel_data->name)
                {
                    zend_string_release(kernel_data->name);
                }
                if (kernel_data->parameters)
                {
                    for (int i = 0; i < kernel_data->parameters->total; i++)
                    {
                        efree(kernel_data->parameters->parameters[i]);
                    }
                    efree(kernel_data->parameters->parameters);
                    efree(kernel_data->parameters);
                }
                efree(kernel_data);
            }
        }
        ZEND_HASH_FOREACH_END();

        zend_hash_destroy(module->kernel_functions);
        efree(module->kernel_functions);
        module->kernel_functions = NULL;
    }

    zval *ptx_b64 = zend_hash_str_find(Z_ARR_P(data), "ptx_b64", 7);
    if (ptx_b64 && Z_TYPE_P(ptx_b64) == IS_STRING)
    {
        zend_string *ptx_zstr = php_base64_decode(
            (unsigned char *)Z_STRVAL_P(ptx_b64),
            Z_STRLEN_P(ptx_b64));

        if (!ptx_zstr)
        {
            zend_throw_exception_ex(NULL, 0, "Failed to decode base64 PTX data");
            return;
        }

        module->ptx_code = estrndup(ZSTR_VAL(ptx_zstr), ZSTR_LEN(ptx_zstr));
        module->ptx_size = ZSTR_LEN(ptx_zstr);

        zend_string_release(ptx_zstr);

        if (module->ptx_size == 0)
        {
            efree(module->ptx_code);
            module->ptx_code = NULL;
            zend_throw_exception_ex(NULL, 0, "Decoded PTX data is empty");
            return;
        }
    }

    zval *kernels_zv = zend_hash_str_find(Z_ARR_P(data), "kernels", sizeof("kernels") - 1);
    if (kernels_zv && Z_TYPE_P(kernels_zv) == IS_ARRAY)
    {
        ALLOC_HASHTABLE(module->kernel_functions);
        zend_hash_init(module->kernel_functions, 8, NULL, NULL, 0);

        zend_string *key;
        zval *kernel_zv;

        ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARR_P(kernels_zv), key, kernel_zv)
        {
            if (Z_TYPE_P(kernel_zv) == IS_ARRAY)
            {
                cuda_kernel_data *kernel_data = (cuda_kernel_data *)emalloc(sizeof(cuda_kernel_data));
                memset(kernel_data, 0, sizeof(cuda_kernel_data));

                zval *name_zv = zend_hash_str_find(Z_ARR_P(kernel_zv), "name", sizeof("name") - 1);
                if (name_zv && Z_TYPE_P(name_zv) == IS_STRING)
                {
                    kernel_data->name = zend_string_init(Z_STRVAL_P(name_zv), Z_STRLEN_P(name_zv), 0);
                }

                zval *params_zv = zend_hash_str_find(Z_ARR_P(kernel_zv), "parameters", sizeof("parameters") - 1);
                if (params_zv && Z_TYPE_P(params_zv) == IS_ARRAY && zend_hash_num_elements(Z_ARR_P(params_zv)) > 0)
                {
                    int param_count = zend_hash_num_elements(Z_ARR_P(params_zv));
                    kernel_data->parameters = (func_parameter_list_t *)emalloc(sizeof(func_parameter_list_t));
                    kernel_data->parameters->total = param_count;
                    kernel_data->parameters->parameters = (func_parameter **)emalloc(sizeof(func_parameter *) * param_count);

                    int i = 0;
                    zval *param_zv;
                    ZEND_HASH_FOREACH_VAL(Z_ARR_P(params_zv), param_zv)
                    {
                        if (Z_TYPE_P(param_zv) == IS_ARRAY)
                        {
                            kernel_data->parameters->parameters[i] = (func_parameter *)emalloc(sizeof(func_parameter));
                            memset(kernel_data->parameters->parameters[i], 0, sizeof(func_parameter));

                            zval *param_name = zend_hash_str_find(Z_ARR_P(param_zv), "name", sizeof("name") - 1);
                            zval *param_type = zend_hash_str_find(Z_ARR_P(param_zv), "type", sizeof("type") - 1);
                            zval *param_dtype = zend_hash_str_find(Z_ARR_P(param_zv), "dtype", sizeof("dtype") - 1);
                            zval *param_second_dtype = zend_hash_str_find(Z_ARR_P(param_zv), "second_dtype", sizeof("second_dtype") - 1);

                            if (param_name && Z_TYPE_P(param_name) == IS_STRING)
                            {
                                strncpy(kernel_data->parameters->parameters[i]->name,
                                        Z_STRVAL_P(param_name),
                                        sizeof(kernel_data->parameters->parameters[i]->name) - 1);
                                kernel_data->parameters->parameters[i]->name[sizeof(kernel_data->parameters->parameters[i]->name) - 1] = '\0';
                            }
                            if (param_type)
                                kernel_data->parameters->parameters[i]->type = zval_get_long(param_type);
                            if (param_dtype)
                                kernel_data->parameters->parameters[i]->dtype = zval_get_long(param_dtype);
                            if (param_second_dtype)
                                kernel_data->parameters->parameters[i]->second_dtype = zval_get_long(param_second_dtype);
                            i++;
                        }
                    }
                    ZEND_HASH_FOREACH_END();
                }
                else
                {
                    kernel_data->parameters = NULL;
                }

                zend_hash_add_ptr(module->kernel_functions, key, kernel_data);
            }
        }
        ZEND_HASH_FOREACH_END();
    }

    module->from_serialize = 1;
    module->cu_context = NULL;
    module->cu_stream = NULL;
    module->cu_device = 0;

    ALLOC_HASHTABLE(module->loaded_modules);
    zend_hash_init(module->loaded_modules, 8, NULL, NULL, 0);

    module->has_pending_operations = 0;

    ALLOC_HASHTABLE(module->async_operations);
    zend_hash_init(module->async_operations, 8, NULL, NULL, 0);
    module->next_async_op_id = 1;

    module->total_memory_allocated = 0;
    module->peak_memory_usage = 0;
    module->kernel_execution_count = 0;
    module->total_execution_time_ms = 0.0;
}

ZEND_METHOD(CompiledModule, getAsyncStatus)
{
    zend_long op_id = -1;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    module_cleanup_timeout_operations(module);

    if (op_id == -1)
    {
        array_init(return_value);

        zend_ulong num_idx;
        cuda_async_operation *op;

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op)
            {
                zval op_info;
                array_init(&op_info);

                add_assoc_long(&op_info, "id", op->id);
                add_assoc_bool(&op_info, "is_active", op->is_active);
                if (op->kernel_name)
                {
                    add_assoc_string(&op_info, "kernel", ZSTR_VAL(op->kernel_name));
                }
                add_assoc_double(&op_info, "elapsed_ms",
                                 module_get_current_time_ms() - op->start_time);
                add_assoc_long(&op_info, "temp_buffers_count", op->temp_buffers_count);

                add_index_zval(return_value, num_idx, &op_info);
            }
        }
        ZEND_HASH_FOREACH_END();
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);

        if (!op)
        {
            RETURN_NULL();
        }

        array_init(return_value);
        add_assoc_long(return_value, "id", op->id);
        add_assoc_bool(return_value, "is_active", op->is_active);
        if (op->kernel_name)
        {
            add_assoc_string(return_value, "kernel", ZSTR_VAL(op->kernel_name));
        }
        add_assoc_double(return_value, "elapsed_ms",
                         module_get_current_time_ms() - op->start_time);
        add_assoc_long(return_value, "temp_buffers_count", op->temp_buffers_count);
    }
}

ZEND_METHOD(CompiledModule, getPendingOperations)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    module_cleanup_timeout_operations(module);

    array_init(return_value);

    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && op->is_active)
        {
            zval op_info;
            array_init(&op_info);

            add_assoc_long(&op_info, "id", op->id);
            if (op->kernel_name)
            {
                add_assoc_string(&op_info, "kernel", ZSTR_VAL(op->kernel_name));
            }
            add_assoc_double(&op_info, "elapsed_ms",
                             module_get_current_time_ms() - op->start_time);

            add_index_zval(return_value, num_idx, &op_info);
        }
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(CompiledModule, getStats)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    array_init(return_value);

    add_assoc_long(return_value, "kernel_execution_count", module->kernel_execution_count);
    add_assoc_double(return_value, "total_execution_time_ms", module->total_execution_time_ms);

    if (module->kernel_execution_count > 0)
    {
        add_assoc_double(return_value, "avg_execution_time_ms",
                         module->total_execution_time_ms / module->kernel_execution_count);
    }

    int pending_count = 0;
    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && op->is_active)
        {
            pending_count++;
        }
    }
    ZEND_HASH_FOREACH_END();

    add_assoc_long(return_value, "pending_operations", pending_count);
    add_assoc_long(return_value, "total_operations", zend_hash_num_elements(module->async_operations));
    add_assoc_bool(return_value, "has_pending_operations", module->has_pending_operations);
    add_assoc_long(return_value, "stream_pool_size", module->stream_pool ? module->stream_pool->size : 0);
    add_assoc_long(return_value, "stream_pool_capacity", module->stream_pool ? module->stream_pool->capacity : 0);

    if (pending_count > 0)
    {
        zval active_ops;
        array_init(&active_ops);

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op && op->is_active)
            {
                zval op_info;
                array_init(&op_info);

                add_assoc_long(&op_info, "id", op->id);
                if (op->kernel_name)
                {
                    add_assoc_string(&op_info, "kernel", ZSTR_VAL(op->kernel_name));
                }
                add_assoc_double(&op_info, "elapsed_ms",
                                 module_get_current_time_ms() - op->start_time);

                add_next_index_zval(&active_ops, &op_info);
            }
        }
        ZEND_HASH_FOREACH_END();

        add_assoc_zval(return_value, "active_operations", &active_ops);
    }
}

ZEND_METHOD(CompiledModule, cancelOperation)
{
    zend_long op_id;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
    if (!op)
    {
        zend_throw_exception_ex(NULL, 0, "Async operation %ld not found", op_id);
        RETURN_FALSE;
    }

    if (!op->is_active)
    {
        RETURN_TRUE;
    }

    CUresult cu_result = cuStreamSynchronize(op->stream);
    if (cu_result != CUDA_SUCCESS && cu_result != CUDA_ERROR_NOT_READY)
    {
        module_log_error("Failed to synchronize stream before cancellation: %s",
                         module_get_cuda_error_string(cu_result));
    }

    op->is_active = 0;

    zend_ulong num_idx;
    cuda_async_operation *check_op;
    module->has_pending_operations = 0;
    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, check_op)
    {
        if (check_op && check_op->is_active)
        {
            module->has_pending_operations = 1;
            break;
        }
    }
    ZEND_HASH_FOREACH_END();

    RETURN_TRUE;
}

ZEND_METHOD(CompiledModule, cleanup)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    zend_ulong num_idx;
    cuda_async_operation *op;
    int cleaned_count = 0;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && !op->is_active)
        {
            module_cleanup_async_operation_by_id(module, op->id);
            cleaned_count++;
        }
    }
    ZEND_HASH_FOREACH_END();

    RETURN_LONG(cleaned_count);
}

static void module_free_object(zend_object *object)
{
    cuda_module_object *module = Z_CUDA_MODULE_FROM_OBJ(object);

    if (module->async_operations)
    {
        module_cleanup_all_async_operations(module);
        zend_hash_destroy(module->async_operations);
        efree(module->async_operations);
        module->async_operations = NULL;
    }

    module_destroy_stream_pool(module);

    module_cleanup_cuda_resources(module);

    if (module->ptx_code)
    {
        efree(module->ptx_code);
        module->ptx_code = NULL;
    }

    if (module->kernel_functions)
    {
        zend_string *key;
        cuda_kernel_data *kernel_data;

        ZEND_HASH_FOREACH_STR_KEY_PTR(module->kernel_functions, key, kernel_data)
        {
            if (kernel_data)
            {
                if (kernel_data->name)
                {
                    zend_string_release(kernel_data->name);
                }
                if (kernel_data->parameters)
                {
                    for (int i = 0; i < kernel_data->parameters->total; i++)
                    {
                        efree(kernel_data->parameters->parameters[i]);
                    }
                    efree(kernel_data->parameters->parameters);
                    efree(kernel_data->parameters);
                }
                efree(kernel_data);
            }
        }
        ZEND_HASH_FOREACH_END();

        zend_hash_destroy(module->kernel_functions);
        efree(module->kernel_functions);
        module->kernel_functions = NULL;
    }

    if (module->functions)
    {
        zend_hash_destroy(module->functions);
        efree(module->functions);
        module->functions = NULL;
    }

    if (module->loaded_modules)
    {
        zend_hash_destroy(module->loaded_modules);
        efree(module->loaded_modules);
        module->loaded_modules = NULL;
    }

    zend_object_std_dtor(&module->std);
}

static zend_object *module_create_object(zend_class_entry *class_type)
{
    cuda_module_object *module = (cuda_module_object *)ecalloc(1, sizeof(cuda_module_object));

    zend_object_std_init(&module->std, class_type);
    module->std.handlers = &module_handlers;

    module->ptx_code = NULL;
    module->ptx_size = 0;
    module->functions = NULL;
    module->kernel_functions = NULL;
    module->cu_device = 0;
    module->cu_context = NULL;
    module->cu_stream = NULL;
    module->stream_pool = NULL;

    ALLOC_HASHTABLE(module->loaded_modules);
    zend_hash_init(module->loaded_modules, 8, NULL, NULL, 0);

    module->from_serialize = 0;

    module->has_pending_operations = 0;
    ALLOC_HASHTABLE(module->async_operations);
    zend_hash_init(module->async_operations, 8, NULL, NULL, 0);
    module->next_async_op_id = 1;

    module->total_memory_allocated = 0;
    module->peak_memory_usage = 0;
    module->kernel_execution_count = 0;
    module->total_execution_time_ms = 0.0;

    return &module->std;
}

int module_init(void)
{
    zend_class_entry ce;

    INIT_CLASS_ENTRY(ce, MODULE_CLASS_NAME, module_methods);
    cuda_module_ce = zend_register_internal_class(&ce);
    cuda_module_ce->create_object = module_create_object;
    cuda_module_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&module_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    module_handlers.offset = XtOffsetOf(cuda_module_object, std);
    module_handlers.free_obj = module_free_object;

    return 1;
}

void module_shutdown(void)
{
    g_cuda_initialized = 0;
    pthread_mutex_destroy(&g_cuda_global_init_mutex);
}