#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "php.h"
#include "cuda_wrapper.h"

#ifdef __cplusplus
extern "C"
{
#endif

    extern zend_class_entry *cuda_array_ce;

    PHP_METHOD(CudaArray, __construct);
    PHP_METHOD(CudaArray, multiply);
    PHP_METHOD(CudaArray, divide);
    PHP_METHOD(CudaArray, add);
    PHP_METHOD(CudaArray, matmul);
    PHP_METHOD(CudaArray, getShape);
    PHP_METHOD(CudaArray, toArray);

    PHP_METHOD(CudaArray, zeros);
    PHP_METHOD(CudaArray, ones);

    void cuda_array_init();

#ifdef __cplusplus
}
#endif

#endif