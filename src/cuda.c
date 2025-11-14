#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "cuda.h"
#include "cuda_wrapper.h"
#include "cuda_arginfo.h"
#include "cuda_array.h"

ZEND_FUNCTION(cuda_get_device_count)
{
    int count = cuda_wrapper_get_device_count();

    if (count < 0)
    {
        php_error_docref(NULL, E_WARNING, "Failed to get device count");
        RETURN_LONG(-1);
    }

    RETURN_LONG(count);
}

ZEND_FUNCTION(cuda_get_device_info)
{
    zend_long device_id = 0;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(device_id)
    ZEND_PARSE_PARAMETERS_END();

    char name[256];
    int major, minor;
    size_t total_mem;

    int error = cuda_wrapper_get_device_properties(device_id, name, sizeof(name),
                                                   &major, &minor, &total_mem);

    if (error != 1)
    {
        php_error_docref(NULL, E_WARNING, "Failed to get device properties");
        RETURN_NULL();
    }

    array_init(return_value);
    add_assoc_string(return_value, "name", name);
    add_assoc_long(return_value, "compute_capability_major", major);
    add_assoc_long(return_value, "compute_capability_minor", minor);
    add_assoc_long(return_value, "total_global_memory", total_mem);
}

ZEND_FUNCTION(cuda_set_device)
{
    zend_long device_id;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(device_id)
    ZEND_PARSE_PARAMETERS_END();

    int success = cuda_wrapper_set_device((int)device_id);

    if (!success)
    {
        php_error_docref(NULL, E_WARNING, "Failed to set device %d", (int)device_id);
        RETURN_FALSE;
    }

    RETURN_TRUE;
}

ZEND_FUNCTION(cuda_get_current_device)
{
    int device = cuda_wrapper_get_current_device();

    if (device == -1)
    {
        php_error_docref(NULL, E_WARNING, "Failed to get current device");
        RETURN_LONG(-1);
    }

    RETURN_LONG(device);
}

ZEND_FUNCTION(cuda_get_memory_info)
{
    size_t free_mem, total_mem;
    int success = cuda_wrapper_get_memory_info(&free_mem, &total_mem);

    if (!success)
    {
        php_error_docref(NULL, E_WARNING, "Failed to get memory info");
        RETURN_NULL();
    }

    array_init(return_value);
    add_assoc_long(return_value, "free_memory", free_mem);
    add_assoc_long(return_value, "total_memory", total_mem);
    add_assoc_long(return_value, "used_memory", total_mem - free_mem);
    add_assoc_double(return_value, "usage_percentage", ((double)(total_mem - free_mem) / total_mem) * 100.0);
}

ZEND_FUNCTION(cuda_device_reset)
{
    int success = cuda_wrapper_device_reset();

    if (!success)
    {
        php_error_docref(NULL, E_WARNING, "Failed to reset device");
        RETURN_FALSE;
    }

    RETURN_TRUE;
}

ZEND_FUNCTION(cuda_get_driver_version)
{
    int driver_version = cuda_wrapper_get_driver_version();

    if (driver_version == -1)
    {
        php_error_docref(NULL, E_WARNING, "Failed to get driver version");
        RETURN_NULL();
    }

    array_init(return_value);
    add_assoc_long(return_value, "version", driver_version);

    char version_str[16];
    snprintf(version_str, sizeof(version_str), "%d.%d", driver_version / 1000, (driver_version % 100) / 10);
    add_assoc_string(return_value, "version_string", version_str);
}

ZEND_FUNCTION(cuda_get_runtime_version)
{
    int runtime_version = cuda_wrapper_get_runtime_version();

    if (runtime_version == -1)
    {
        php_error_docref(NULL, E_WARNING, "Failed to get runtime version");
        RETURN_NULL();
    }

    array_init(return_value);
    add_assoc_long(return_value, "version", runtime_version);

    char version_str[16];
    snprintf(version_str, sizeof(version_str), "%d.%d", runtime_version / 1000, (runtime_version % 100) / 10);
    add_assoc_string(return_value, "version_string", version_str);
}

ZEND_FUNCTION(cuda_synchronize)
{
    int success = cuda_wrapper_synchronize();

    if (!success)
    {
        php_error_docref(NULL, E_WARNING, "Failed to synchronize device");
        RETURN_FALSE;
    }

    RETURN_TRUE;
}


ZEND_FUNCTION(cuda_get_last_error)
{
    int error = cuda_wrapper_error();
    if (error == 0)
    {
        RETURN_NULL();
    }

    array_init(return_value);
    add_assoc_long(return_value, "code", 1);
    add_assoc_string(return_value, "error_message", cuda_wrapper_get_error_string(error));
    add_assoc_string(return_value, "error_type", cuda_wrapper_get_error_type(error));
}

ZEND_FUNCTION(cuda_clear_error)
{
    cuda_wrapper_error();
    RETURN_TRUE;
}

ZEND_FUNCTION(cuda_get_peer_access)
{
    zend_long device1, device2;

    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_LONG(device1)
    Z_PARAM_LONG(device2)
    ZEND_PARSE_PARAMETERS_END();

    int result = cuda_wrapper_get_peer_access((int)device1, (int)device2);

    if (result == -1)
    {
        php_error_docref(NULL, E_WARNING, "Failed to check peer access");
        RETURN_NULL();
    }

    RETURN_BOOL(result);
}

static zend_function_entry cuda_functions[] = {
    PHP_FE(cuda_get_device_count, arginfo_cuda_get_device_count)
    PHP_FE(cuda_get_device_info, arginfo_cuda_get_device_info)
    PHP_FE(cuda_set_device, arginfo_cuda_set_device)
    PHP_FE(cuda_get_current_device, arginfo_cuda_get_current_device)
    PHP_FE(cuda_get_memory_info, arginfo_cuda_get_memory_info)
    PHP_FE(cuda_device_reset, arginfo_cuda_device_reset)
    PHP_FE(cuda_synchronize, arginfo_cuda_synchronize)
    PHP_FE(cuda_get_driver_version, arginfo_cuda_get_driver_version)
    PHP_FE(cuda_get_runtime_version, arginfo_cuda_get_runtime_version)
    PHP_FE(cuda_get_last_error, arginfo_cuda_get_last_error)
    PHP_FE(cuda_clear_error, arginfo_cuda_clear_error)
    PHP_FE(cuda_get_peer_access, arginfo_cuda_get_peer_access)
    PHP_FE_END
};

PHP_MINIT_FUNCTION(cuda)
{
    int count = cuda_wrapper_get_device_count();
    if (count < 0)
    {
        php_error_docref(NULL, E_WARNING, "CUDA initialization failed");
    }
    
    cuda_array_init();

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(cuda)
{
    cuda_wrapper_device_reset();
    return SUCCESS;
}

zend_module_entry cuda_module_entry = {
    STANDARD_MODULE_HEADER,
    PHP_CUDA_EXTNAME,
    cuda_functions,
    PHP_MINIT(cuda),
    PHP_MSHUTDOWN(cuda),
    NULL,
    NULL,
    NULL,
    PHP_CUDA_VERSION,
    STANDARD_MODULE_PROPERTIES};

#ifdef COMPILE_DL_CUDA
ZEND_GET_MODULE(cuda)
#endif