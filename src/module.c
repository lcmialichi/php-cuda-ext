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
#include <time.h>
#include "ext/standard/php_standard.h"

extern zend_class_entry *cuda_array_ce;

zend_class_entry *cuda_module_ce;
static zend_object_handlers module_handlers;
static zend_bool g_cuda_initialized = 0;

static zend_object *module_create_object(zend_class_entry *class_type);
static void module_free_object(zend_object *object);
static zend_bool module_initialize_cuda_context(cuda_module_object *module);
static void module_cleanup_cuda_resources(cuda_module_object *module);

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
    case CUDA_ERROR_PROFILER_DISABLED:
        return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
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

static zend_bool module_initialize_cuda_context(cuda_module_object *module)
{
    CUresult cu_result;

    if (!g_cuda_initialized)
    {
        cu_result = cuInit(0);
        if (cu_result != CUDA_SUCCESS)
        {
            zend_throw_exception_ex(NULL, 0, "Failed to initialize CUDA: %s",
                                    module_get_cuda_error_string(cu_result));
            return 0;
        }
        g_cuda_initialized = 1;
    }

    if (module->cu_context)
    {
        return 1;
    }

    cu_result = cuDeviceGet(&module->cu_device, 0);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0, "Failed to get CUDA device: %s",
                                module_get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuCtxCreate(&module->cu_context, CU_CTX_SCHED_AUTO, module->cu_device);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0, "Failed to create CUDA context: %s",
                                module_get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuStreamCreate(&module->cu_stream, CU_STREAM_DEFAULT);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0, "Failed to create CUDA stream: %s",
                                module_get_cuda_error_string(cu_result));
        cuCtxDestroy(module->cu_context);
        module->cu_context = NULL;
        return 0;
    }

    if (!module->loaded_modules)
    {
        ALLOC_HASHTABLE(module->loaded_modules);
        zend_hash_init(module->loaded_modules, 8, NULL, NULL, 0);
    }

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
        module->cu_stream = NULL;
    }

    if (module->cu_context)
    {
        cu_result = cuCtxDestroy(module->cu_context);
        module->cu_context = NULL;
    }
}

static CUmodule module_get_or_load_module(cuda_module_object *module, zend_string *kernel_name)
{
    CUresult cu_result;
    CUmodule *cached_module = (CUmodule *)zend_hash_find_ptr(module->loaded_modules, kernel_name);
    if (cached_module)
    {
        return *cached_module;
    }

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
    zend_hash_add_ptr(module->loaded_modules, kernel_name, module_ptr);

    return cu_module;
}

static zend_bool module_validate_launch_config(CUdevice cu_device, int grid[3], int block[3])
{
    int max_threads, max_block_dims[3], max_grid_dims[3];
    CUresult cu_result;

    cu_result = cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cu_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_block_dims[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, cu_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_block_dims[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, cu_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_block_dims[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, cu_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_grid_dims[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, cu_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_grid_dims[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, cu_device);
    if (cu_result != CUDA_SUCCESS)
        return 0;

    cu_result = cuDeviceGetAttribute(&max_grid_dims[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, cu_device);
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

static zend_bool module_prepare_cuda_arguments(cuda_kernel_data *kernel, zval *args, int argc,
                                               void ***cuda_args_ptr, tensor_t ***tensors_ptr,
                                               int *tensors_count_ptr, void ***temp_buffers_ptr,
                                               int *temp_buffers_count_ptr)
{
    void **cuda_args = (void **)emalloc(sizeof(void *) * argc);
    tensor_t **tensors_to_sync = (tensor_t **)emalloc(sizeof(tensor_t *) * argc);
    int tensors_count = 0;
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
                if (param->type == INPUT)
                {
                    tensors_to_sync[tensors_count++] = tensor;
                }
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
            tensors_to_sync[tensors_count++] = tensor;
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
        *tensors_ptr = tensors_to_sync;
        *tensors_count_ptr = tensors_count;
        *temp_buffers_ptr = temp_gpu_buffers;
        *temp_buffers_count_ptr = temp_buffers_count;
    }
    else
    {
        for (int i = 0; i < temp_buffers_count; i++)
        {
            efree(temp_gpu_buffers[i]);
        }
        efree(cuda_args);
        efree(tensors_to_sync);
        efree(temp_gpu_buffers);
    }

    return valid;
}

static zend_bool module_execute_cuda_kernel(cuda_module_object *module,
                                            cuda_kernel_data *kernel,
                                            int grid[3], int block[3],
                                            void **cuda_args, int argc)
{
    CUresult cu_result;
    CUfunction cu_function;

    if (!module_initialize_cuda_context(module))
    {
        return 0;
    }

    if (!module_validate_launch_config(module->cu_device, grid, block))
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
                               module->cu_stream,
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

    int grid[3] = {1, 1, 1};
    int block[3] = {1, 1, 1};

    if (config_zv && Z_TYPE_P(config_zv) == IS_ARRAY)
    {
        zval *grid_zv = zend_hash_str_find(Z_ARR_P(config_zv), "grid", sizeof("grid") - 1);
        zval *block_zv = zend_hash_str_find(Z_ARR_P(config_zv), "block", sizeof("block") - 1);

        if (!grid_zv)
            grid_zv = zend_hash_index_find(Z_ARR_P(config_zv), 0);
        if (!block_zv)
            block_zv = zend_hash_index_find(Z_ARR_P(config_zv), 1);

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

    void **cuda_args = NULL;
    tensor_t **tensors_to_sync = NULL;
    int tensors_count = 0;
    void **temp_gpu_buffers = NULL;
    int temp_buffers_count = 0;

    zend_bool args_prepared = module_prepare_cuda_arguments(kernel, args, argc,
                                                            &cuda_args, &tensors_to_sync,
                                                            &tensors_count, &temp_gpu_buffers,
                                                            &temp_buffers_count);
    if (!args_prepared)
    {
        if (args)
            efree(args);
        RETURN_FALSE;
    }

    zend_bool success = module_execute_cuda_kernel(module, kernel, grid, block, cuda_args, argc);
    CUresult cu_result = cuStreamSynchronize(module->cu_stream);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to synchronize stream: %s",
                                module_get_cuda_error_string(cu_result));
        return;
    }

    for (int i = 0; i < temp_buffers_count; i++)
    {
        efree(temp_gpu_buffers[i]);
    }
    if (cuda_args)
        efree(cuda_args);
    if (tensors_to_sync)
        efree(tensors_to_sync);
    if (temp_gpu_buffers)
        efree(temp_gpu_buffers);
    if (args)
        efree(args);

    RETURN_BOOL(success);
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

    int grid[3] = {1, 1, 1};
    int block[3] = {1, 1, 1};

    if (config_zv && Z_TYPE_P(config_zv) == IS_ARRAY)
    {
        zval *grid_zv = zend_hash_str_find(Z_ARR_P(config_zv), "grid", sizeof("grid") - 1);
        zval *block_zv = zend_hash_str_find(Z_ARR_P(config_zv), "block", sizeof("block") - 1);

        if (!grid_zv)
            grid_zv = zend_hash_index_find(Z_ARR_P(config_zv), 0);
        if (!block_zv)
            block_zv = zend_hash_index_find(Z_ARR_P(config_zv), 1);

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

    void **cuda_args = NULL;
    tensor_t **tensors_to_sync = NULL;
    int tensors_count = 0;
    void **temp_gpu_buffers = NULL;
    int temp_buffers_count = 0;

    zend_bool args_prepared = module_prepare_cuda_arguments(kernel, args, argc,
                                                            &cuda_args, &tensors_to_sync,
                                                            &tensors_count, &temp_gpu_buffers,
                                                            &temp_buffers_count);
    if (!args_prepared)
    {
        if (args)
            efree(args);
        RETURN_FALSE;
    }

    zend_bool success = module_execute_cuda_kernel(module, kernel, grid, block, cuda_args, argc);
    for (int i = 0; i < temp_buffers_count; i++)
    {
        efree(temp_gpu_buffers[i]);
    }
    if (cuda_args)
        efree(cuda_args);
    if (tensors_to_sync)
        efree(tensors_to_sync);
    if (temp_gpu_buffers)
        efree(temp_gpu_buffers);
    if (args)
        efree(args);

    if (success)
    {
        module->has_pending_operations = 1;
    }

    RETURN_BOOL(success);
}

ZEND_METHOD(CompiledModule, sync)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    CUresult cu_result;

    if (!module->cu_stream)
    {
        RETURN_TRUE;
    }

    cu_result = cuStreamSynchronize(module->cu_stream);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to synchronize stream: %s",
                                module_get_cuda_error_string(cu_result));
        RETURN_FALSE;
    }

    module->has_pending_operations = 0;

    RETURN_TRUE;
}

ZEND_METHOD(CompiledModule, isFinished)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    CUresult cu_result;

    if (!module->cu_stream)
    {
        RETURN_TRUE;
    }

    cu_result = cuStreamQuery(module->cu_stream);

    if (cu_result == CUDA_SUCCESS)
    {
        module->has_pending_operations = 0;
        RETURN_TRUE;
    }
    else if (cu_result == CUDA_ERROR_NOT_READY)
    {
        RETURN_FALSE;
    }
    else
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to query stream: %s",
                                module_get_cuda_error_string(cu_result));
        RETURN_FALSE;
    }
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
    module->loaded_modules = NULL;
}

static void module_free_object(zend_object *object)
{
    cuda_module_object *module = Z_CUDA_MODULE_FROM_OBJ(object);

    module_cleanup_cuda_resources(module);

    if (module->ptx_code)
    {
        efree(module->ptx_code);
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
    }

    if (module->functions)
    {
        zend_hash_destroy(module->functions);
        efree(module->functions);
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
    module->loaded_modules = NULL;
    module->cu_device = 0;
    module->cu_context = NULL;
    module->cu_stream = NULL;
    module->from_serialize = 0;
    module->has_pending_operations = 0;

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
}