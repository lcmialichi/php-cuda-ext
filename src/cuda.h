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

ZEND_FUNCTION(cuda_get_device_count);
ZEND_FUNCTION(cuda_get_device_info);
ZEND_FUNCTION(cuda_set_device);
ZEND_FUNCTION(cuda_get_current_device);
ZEND_FUNCTION(cuda_get_memory_info);
ZEND_FUNCTION(cuda_device_reset);
ZEND_FUNCTION(cuda_get_driver_version);
ZEND_FUNCTION(cuda_get_runtime_version);
ZEND_FUNCTION(cuda_get_last_error);
ZEND_FUNCTION(cuda_clear_error);
ZEND_FUNCTION(cuda_get_peer_access);
ZEND_FUNCTION(cuda_synchronize);

PHP_MINIT_FUNCTION(cuda);
PHP_MSHUTDOWN_FUNCTION(cuda);

#endif