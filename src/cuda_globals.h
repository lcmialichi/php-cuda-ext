#ifndef CUDA_GLOBALS_H
#define CUDA_GLOBALS_H

#include "php.h"
#include "config.h"

#ifdef ZTS
#include "TSRM.h"
#endif

#include "cuda.h"

ZEND_EXTERN_MODULE_GLOBALS(cuda);

#ifdef ZTS
    #define CUDA_G(v) TSRMG(cuda_globals_id, zend_cuda_globals *, v)
#else
    extern zend_cuda_globals cuda_globals;
    #define CUDA_G(v) (cuda_globals.v)
#endif

#endif