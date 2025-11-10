#ifndef CUDA_ARRAY_ARGINFO_H
#define CUDA_ARRAY_ARGINFO_H

#include "php.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_multiply, 0, 1, CudaArray, 0)
    ZEND_ARG_INFO(0, other)  // Aceita CudaArray, float, int, etc.
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_matmul, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, other, CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_getShape, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_transpose, 0, 0, 0)
ZEND_END_ARG_INFO()


ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_toArray, 0, 0, 0)
ZEND_END_ARG_INFO()


#endif