#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H
#endif

#include "php.h"
#include "cuda_wrapper.h"
#include "tensor.h"

extern zend_class_entry *cuda_array_ce;

typedef struct
{
    tensor_t *tensor_handle;
    zend_array *shape;
    zend_object obj;
} cuda_array_obj;


void cuda_array_init();

ZEND_METHOD(CudaArray, __construct);
ZEND_METHOD(CudaArray, __invoke);

ZEND_METHOD(CudaArray, zeros);
ZEND_METHOD(CudaArray, ones);
ZEND_METHOD(CudaArray, full);

ZEND_METHOD(CudaArray, matmul);
ZEND_METHOD(CudaArray, transpose);

ZEND_METHOD(CudaArray, multiply);
ZEND_METHOD(CudaArray, divide);
ZEND_METHOD(CudaArray, add);
ZEND_METHOD(CudaArray, subtract);
ZEND_METHOD(CudaArray, power);

ZEND_METHOD(CudaArray, sqrt);
ZEND_METHOD(CudaArray, exp);
ZEND_METHOD(CudaArray, log);
ZEND_METHOD(CudaArray, sin);
ZEND_METHOD(CudaArray, cos);
ZEND_METHOD(CudaArray, tan);
ZEND_METHOD(CudaArray, abs);
ZEND_METHOD(CudaArray, neg);

ZEND_METHOD(CudaArray, greater);
ZEND_METHOD(CudaArray, less);
ZEND_METHOD(CudaArray, equal);
ZEND_METHOD(CudaArray, notEqual);
ZEND_METHOD(CudaArray, greaterEqual);
ZEND_METHOD(CudaArray, lessEqual);

ZEND_METHOD(CudaArray, reshape);
ZEND_METHOD(CudaArray, flatten);

ZEND_METHOD(CudaArray, getShape);
ZEND_METHOD(CudaArray, getStrides);
ZEND_METHOD(CudaArray, toArray);
