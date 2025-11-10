#ifndef CUDA_ARRAY_WRAPPER_H
#define CUDA_ARRAY_WRAPPER_H

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "php.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* data;
    int* shape;
    int ndims;
    size_t total_size;
    cudnnTensorDescriptor_t desc;
} tensor_t;

typedef enum {
    CUDNN_OP_ADD,
    CUDNN_OP_MUL,
    CUDNN_OP_SUB,
    CUDNN_OP_DIV,
    CUDNN_OP_MIN,
    CUDNN_OP_MAX,
    KERNEL_DIV
} cuda_op_type_t;


int cuda_wrapper_init();

tensor_t* cuda_tensor_create(const int shape[], int ndims, const float data[]);
tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims);
tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, float value);
tensor_t* cuda_tensor_create_empty(const int shape[], int ndims);

cudaError_t cuda_flatten_php_array_to_gpu(zval *data, float *gpu_data, int *index, size_t total_size);
static void flatten_php_array_to_buffer(zval *data, float *buffer, int *index);

void cuda_tensor_destroy(tensor_t* tensor);

// operations
tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, tensor_t *result, cuda_op_type_t op_type);
tensor_t *cuda_tensor_add(tensor_t *a, tensor_t *b);
tensor_t* cuda_tensor_multiply(tensor_t* a, tensor_t* b);
tensor_t *cuda_tensor_divide(tensor_t *a, tensor_t *b);
tensor_t* cuda_tensor_matmul(tensor_t* a, tensor_t* b);

int* cuda_tensor_get_shape(tensor_t* tensor);
tensor_t *cuda_tensor_transpose(tensor_t *tensor);

tensor_t *cuda_tensor_copy(tensor_t *tensor);
size_t cuda_tensor_size(tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif