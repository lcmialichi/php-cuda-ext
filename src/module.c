#include "module.h"
#include "php.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "module_arginfo.h"
#include "kernel_types.h"
#include "tensor.h"
#include <nvrtc.h>
#include <cuda.h>
#include "ca_struct.h"

zend_class_entry *cuda_module_ce;

extern zend_class_entry *cuda_array_ce;

static zend_object_handlers module_handlers;

static zend_object *module_create_object(zend_class_entry *class_type);
static void module_free_object(zend_object *object);

#ifndef Z_CUDA_ARRAY_P
#define Z_CUDA_ARRAY_P(zv) ((cuda_array_obj *)((char *)Z_OBJ_P(zv) - XtOffsetOf(cuda_array_obj, obj)))
#endif

static const char *get_cuda_error_string(CUresult result)
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

static const char *dtype_to_string(dtype_t dtype)
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

static zend_bool validate_cuda_launch_config(CUdevice cu_device, int grid[3], int block[3])
{
    int max_threads, max_block_dims[3], max_grid_dims[3];

    cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cu_device);
    cuDeviceGetAttribute(&max_block_dims[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, cu_device);
    cuDeviceGetAttribute(&max_block_dims[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, cu_device);
    cuDeviceGetAttribute(&max_block_dims[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, cu_device);
    cuDeviceGetAttribute(&max_grid_dims[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, cu_device);
    cuDeviceGetAttribute(&max_grid_dims[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, cu_device);
    cuDeviceGetAttribute(&max_grid_dims[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, cu_device);

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

static void extract_launch_config(zval *grid_zv, zval *block_zv,
                                  cuda_kernel_data *kernel,
                                  int *grid, int *block)
{
    grid[0] = kernel->grid[0];
    grid[1] = kernel->grid[1];
    grid[2] = kernel->grid[2];
    block[0] = kernel->block[0];
    block[1] = kernel->block[1];
    block[2] = kernel->block[2];

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

static zend_bool prepare_cuda_arguments(cuda_kernel_data *kernel, zval *args, int argc,
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

                cuda_array_obj *array_obj = Z_CUDA_ARRAY_P(arg);
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
                    const char *expected = dtype_to_string(param->second_dtype);
                    const char *actual = dtype_to_string(tensor->dtype);
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
                void *scalar_value = NULL;
                switch (param->dtype)
                {
                case INT32:
                {
                    zend_long value = zval_get_long(arg);
                    int *int_ptr = (int *)emalloc(sizeof(int));
                    *int_ptr = (int)value;
                    cuda_args[i] = int_ptr;
                    temp_gpu_buffers[temp_buffers_count++] = int_ptr;
                    break;
                }
                case FLOAT32:
                {
                    double value = zval_get_double(arg);
                    float *float_ptr = (float *)emalloc(sizeof(float));
                    *float_ptr = (float)value;
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
                                        "Output argument %d '%s' must be a CudaArray (passed by reference)",
                                        i + 1, param->name);
                valid = 0;
                break;
            }

            cuda_array_obj *array_obj = Z_CUDA_ARRAY_P(arg);
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
                const char *expected = dtype_to_string(param->second_dtype);
                const char *actual = dtype_to_string(tensor->dtype);
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

static zend_bool execute_cuda_kernel(cuda_module_object *module,
                                     cuda_kernel_data *kernel,
                                     int grid[3], int block[3],
                                     void **cuda_args, int argc)
{
    CUresult cu_result;
    CUdevice cu_device;
    CUcontext cu_context;
    CUmodule cu_module;
    CUfunction cu_function;
    CUstream cu_stream = 0;

    cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to initialize CUDA: %s",
                                get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuDeviceGet(&cu_device, 0);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to get CUDA device: %s",
                                get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuCtxCreate(&cu_context, 0, cu_device);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to create CUDA context: %s",
                                get_cuda_error_string(cu_result));
        return 0;
    }

    if (!validate_cuda_launch_config(cu_device, grid, block))
    {
        zend_throw_exception_ex(NULL, 0,
                                "Invalid grid/block configuration");
        cuCtxDestroy(cu_context);
        return 0;
    }

    cu_result = cuModuleLoadDataEx(&cu_module, module->ptx_code, 0, NULL, NULL);
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to load PTX module: %s",
                                get_cuda_error_string(cu_result));
        cuCtxDestroy(cu_context);
        return 0;
    }

    cu_result = cuModuleGetFunction(&cu_function, cu_module, ZSTR_VAL(kernel->name));
    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to get kernel function '%s': %s",
                                ZSTR_VAL(kernel->name),
                                get_cuda_error_string(cu_result));
        cuModuleUnload(cu_module);
        cuCtxDestroy(cu_context);
        return 0;
    }

    php_printf("DEBUG: Launching kernel '%s'\n", ZSTR_VAL(kernel->name));
    php_printf("DEBUG: Grid: [%d, %d, %d]\n", grid[0], grid[1], grid[2]);
    php_printf("DEBUG: Block: [%d, %d, %d]\n", block[0], block[1], block[2]);
    php_printf("DEBUG: Total threads: %d\n",
               grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2]);

    cu_result = cuLaunchKernel(cu_function,
                               grid[0], grid[1], grid[2],
                               block[0], block[1], block[2],
                               0,
                               cu_stream,
                               cuda_args,
                               NULL);

    php_printf("DEBUG: cuLaunchKernel result: %s\n",
               get_cuda_error_string(cu_result));

    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to launch kernel '%s': %s",
                                ZSTR_VAL(kernel->name),
                                get_cuda_error_string(cu_result));
        cuModuleUnload(cu_module);
        cuCtxDestroy(cu_context);
        return 0;
    }

    cu_result = cuStreamSynchronize(cu_stream);
    php_printf("DEBUG: cuStreamSynchronize result: %s\n",
               get_cuda_error_string(cu_result));

    if (cu_result != CUDA_SUCCESS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Failed to synchronize stream: %s",
                                get_cuda_error_string(cu_result));
        cuModuleUnload(cu_module);
        cuCtxDestroy(cu_context);
        return 0;
    }
    cuModuleUnload(cu_module);
    cuCtxDestroy(cu_context);

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

    php_printf("DEBUG: Kernel: %s\n", ZSTR_VAL(kernel_name));
    php_printf("DEBUG: Config provided: %s\n", config_zv ? "yes" : "no");
    php_printf("DEBUG: Args provided: %s\n", args_zv ? "yes" : "no");

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    cuda_kernel_data *kernel = zend_hash_find_ptr(module->kernel_functions, kernel_name);
    if (!kernel)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel '%s' not found in compiled module",
                                ZSTR_VAL(kernel_name));
        RETURN_FALSE;
    }

    zval *args = NULL;
    int argc = 0;
    
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
        if (args) efree(args);
        RETURN_FALSE;
    }

    zval *grid_zv = NULL;
    zval *block_zv = NULL;
    
    if (config_zv && Z_TYPE_P(config_zv) == IS_ARRAY)
    {
        grid_zv = zend_hash_str_find(Z_ARR_P(config_zv), "grid", sizeof("grid")-1);
        block_zv = zend_hash_str_find(Z_ARR_P(config_zv), "block", sizeof("block")-1);
        
        if (!grid_zv) grid_zv = zend_hash_index_find(Z_ARR_P(config_zv), 0);
        if (!block_zv) block_zv = zend_hash_index_find(Z_ARR_P(config_zv), 1);
    }

    int grid[3], block[3];
    extract_launch_config(grid_zv, block_zv, kernel, grid, block);

    php_printf("DEBUG: Using grid=[%d,%d,%d], block=[%d,%d,%d]\n",
               grid[0], grid[1], grid[2], block[0], block[1], block[2]);

    void **cuda_args = NULL;
    tensor_t **tensors_to_sync = NULL;
    int tensors_count = 0;
    void **temp_gpu_buffers = NULL;
    int temp_buffers_count = 0;

    zend_bool args_prepared = prepare_cuda_arguments(kernel, args, argc,
                                                     &cuda_args, &tensors_to_sync,
                                                     &tensors_count, &temp_gpu_buffers,
                                                     &temp_buffers_count);

    if (!args_prepared)
    {
        if (args) efree(args);
        RETURN_FALSE;
    }

    zend_bool success = execute_cuda_kernel(module, kernel, grid, block, cuda_args, argc);

    for (int i = 0; i < temp_buffers_count; i++)
    {
        efree(temp_gpu_buffers[i]);
    }

    if (cuda_args) efree(cuda_args);
    if (tensors_to_sync) efree(tensors_to_sync);
    if (temp_gpu_buffers) efree(temp_gpu_buffers);
    if (args) efree(args);

    RETURN_BOOL(success);
}

ZEND_METHOD(CompiledModule, hasKernel)
{
    zend_string *name;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (zend_hash_exists(module->kernel_functions, name))
    {
        RETURN_TRUE;
    }
    else
    {
        RETURN_FALSE;
    }
}

ZEND_METHOD(CompiledModule, getKernels)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    array_init(return_value);

    zend_string *key;
    zval *val;

    ZEND_HASH_FOREACH_STR_KEY_VAL(module->kernel_functions, key, val)
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

    if (written == module->ptx_size)
    {
        RETURN_TRUE;
    }
    else
    {
        php_error_docref(NULL, E_WARNING, "Failed to write all bytes to file");
        RETURN_FALSE;
    }
}

static void module_free_object(zend_object *object)
{
    cuda_module_object *module = Z_CUDA_MODULE_FROM_OBJ(object);

    if (module->ptx_code)
    {
        efree(module->ptx_code);
    }

    if (module->functions)
    {
        zend_hash_destroy(module->functions);
        efree(module->functions);
    }

    if (module->kernel_functions)
    {
        zend_hash_destroy(module->kernel_functions);
        efree(module->kernel_functions);
    }

    zend_object_std_dtor(&module->std);
}

static zend_object *module_create_object(zend_class_entry *class_type)
{
    cuda_module_object *module =
        (cuda_module_object *)ecalloc(1, sizeof(cuda_module_object));

    zend_object_std_init(&module->std, class_type);
    module->std.handlers = &module_handlers;

    module->ptx_code = NULL;
    module->ptx_size = 0;
    module->functions = NULL;
    module->kernel_functions = NULL;

    return &module->std;
}

int module_init()
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