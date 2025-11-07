#ifndef PHP_CUDA_H
#define PHP_CUDA_H

#include "php.h"
#include <cuda_runtime.h>

#define PHP_CUDA_VERSION "1.0.0"
#define PHP_CUDA_EXTNAME "cuda"

extern zend_module_entry cuda_module_entry;
#define phpext_cuda_ptr &cuda_module_entry

#ifdef ZTS
#include "TSRM.h"
#endif

PHP_FUNCTION(cuda_get_device_count);
PHP_FUNCTION(cuda_get_device_info);
PHP_FUNCTION(cuda_set_device);
PHP_FUNCTION(cuda_get_current_device);
PHP_FUNCTION(cuda_get_memory_info);
PHP_FUNCTION(cuda_device_reset);
PHP_FUNCTION(cuda_get_driver_version);
PHP_FUNCTION(cuda_get_runtime_version);
PHP_FUNCTION(cuda_get_last_error);
PHP_FUNCTION(cuda_clear_error);
PHP_FUNCTION(cuda_get_peer_access);
PHP_FUNCTION(cuda_synchronize);

PHP_MINIT_FUNCTION(cuda);
PHP_MSHUTDOWN_FUNCTION(cuda);

#endif