
#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include "php.h"
#include "zend_compile.h"
#include "data_types.h"
#include <nvrtc.h>
#include <cuda.h>

typedef struct
{
    zend_string *name;
    dtype_t dtype;
    dtype_t second_dtype;
    int array_dimensions;
    int level;
    enum
    {
        VAR_LOCAL,
        VAR_LOCAL_SHARED
    } var_type;

} local_variable_t;

typedef struct
{
    const char *cuda_name;
    dtype_t return_type;
} cuda_function_match_t;

typedef struct _kernel_obj
{
    zend_object obj;
} kernel_obj;

typedef enum
{
    INPUT,
    OUTPUT,
    PARAMETER
} parameter_type_t;

typedef struct
{
    char name[32];
    dtype_t dtype;
    dtype_t second_dtype;
    parameter_type_t type;
} func_parameter;

typedef struct
{
    int total;
    func_parameter **parameters;
} func_parameter_list_t;

typedef struct
{
    zend_string *filename;
    uint32_t start_line;
    uint32_t end_line;
    zend_string *method_name;
    zend_function *fptr;
} method_source_info_t;

typedef enum
{
    FN_KERNEL,
    FN_DEVICE,
    FN_GLOBAL

} cuda_fn_type;

typedef struct
{
    enum
    {
        CUDA_OBJ_NONE,
        CUDA_OBJ_CUDA,
        CUDA_OBJ_MATH,
        CUDA_OBJ_ATOMIC,
        CUDA_OBJ_SYNC,
        CUDA_OBJ_WARP,
        CUDA_OBJ_THREADIDX,
        CUDA_OBJ_BLOCKIDX,
        CUDA_OBJ_BLOCKDIM,
        CUDA_OBJ_GRIDDIM
    } current_cuda_object;
    HashTable *headers;
    cuda_fn_type fn_type;
    zend_string *name;
    func_parameter_list_t *parameters;
    HashTable local_variables;
    HashTable shared_memory_vars;
    smart_string *cuda_code_buffer;
    int dim_access;
    dtype_t last_evaluated_first_dtype;
    dtype_t last_evaluated_second_dtype;
    dtype_t return_dtype;
    int loop_depth;
    int uses_shared_memory;
    int uses_static_shared_memory;
    int shared_memory_declared;
    int current_line;
} cuda_compilation_context_t;

typedef struct _cached_ptx
{
    char *ptx;
    size_t ptx_size;
    time_t timestamp;
} cached_ptx_t;

typedef struct _cuda_compiler_object
{
    zend_object std;
    char *target_device;
    int optimization_level;
    zend_bool debug_mode;
    zend_bool fast_math;
    HashTable *headers;
    HashTable *kernels;
    HashTable *devices;
    HashTable *ptx_cache;
} cuda_compiler_object;

typedef struct _cuda_kernel_data
{
    zend_string *name;
    zend_string *target;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zend_string *source_code;
    char *cuda_code;
    HashTable *used_devices;
    func_parameter_list_t *parameters;
} cuda_kernel_data;

typedef struct _cuda_device_object
{
    zend_object std;
    zend_string *name;
    zend_string *target;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
} cuda_device_object;

typedef struct _cuda_async_operation
{
    int id;
    zend_string *kernel_name;
    void **cuda_args;
    void **temp_buffers;
    int temp_buffers_count;
    zend_bool is_active;
    double start_time;

    CUstream stream;
    CUevent start_event;
    CUevent end_event;

    int grid[3];
    int block[3];
    int argc;

    CUmodule *cu_module_cache;

    zend_bool owns_module;

    CUresult last_error;
    char error_message[256];

    struct _cuda_async_operation *next;
    struct _cuda_async_operation *prev;
} cuda_async_operation;

typedef struct _cuda_module_object
{
    zend_object std;
    char *ptx_code;
    size_t ptx_size;
    HashTable *functions;
    HashTable *kernel_functions;

    CUdevice cu_device;
    CUcontext cu_context;
    CUstream cu_stream;

    HashTable *loaded_modules;
    int from_serialize;

    zend_bool has_pending_operations;
    int next_async_op_id;
    HashTable *async_operations;
    size_t total_memory_allocated;
    size_t peak_memory_usage;
    int kernel_execution_count;
    double total_execution_time_ms;

    CUstream *stream_pool;
    int stream_pool_size;
    int stream_pool_capacity;
    pthread_mutex_t stream_pool_mutex;

    HashTable *module_cache;

} cuda_module_object;

#define Z_CUDA_DEVICE_P(zv) ((cuda_device_object *)((char *)Z_OBJ_P(zv) - XtOffsetOf(cuda_device_object, std)))
#define Z_CUDA_DEVICE_FROM_OBJ(obj) ((cuda_device_object *)((char *)(obj) - XtOffsetOf(cuda_device_object, std)))
#define Z_CUDA_MODULE_P(zv) ((cuda_module_object *)((char *)Z_OBJ_P(zv) - XtOffsetOf(cuda_module_object, std)))
#define Z_CUDA_MODULE_FROM_OBJ(obj) ((cuda_module_object *)((char *)(obj) - XtOffsetOf(cuda_module_object, std)))
#define Z_CUDA_COMPILER_P(zv) ((cuda_compiler_object *)((char *)Z_OBJ_P(zv) - XtOffsetOf(cuda_compiler_object, std)))
#define Z_CUDA_COMPILER_FROM_OBJ(obj) ((cuda_compiler_object *)((char *)(obj) - XtOffsetOf(cuda_compiler_object, std)))

#endif