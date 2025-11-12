#ifndef CUDA_ARRAY_WRAPPER_H
#define CUDA_ARRAY_WRAPPER_H

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "php.h"
#include "conditional.h"
#include "cuda_kernels.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cuda_flatten_php_array_to_gpu(zval *data, float *gpu_data, int *index, size_t total_size);
static void flatten_php_array_to_buffer(zval *data, float *buffer, int *index);
tensor_t* cuda_tensor_reshape(tensor_t *original, int *new_shape, int new_ndims);   

tensor_t *cuda_tensor_add(tensor_t *a, tensor_t *b);
tensor_t *cuda_tensor_subtract(tensor_t *a, tensor_t *b);
tensor_t* cuda_tensor_multiply(tensor_t* a, tensor_t* b);
tensor_t *cuda_tensor_divide(tensor_t *a, tensor_t *b);
tensor_t* cuda_tensor_matmul(tensor_t* a, tensor_t* b);
tensor_t *cuda_tensor_power(tensor_t *a, tensor_t *b);
tensor_t *cuda_tensor_sqrt(tensor_t *tensor);
tensor_t *cuda_tensor_exp(tensor_t *tensor);
tensor_t *cuda_tensor_log(tensor_t *tensor);
tensor_t *cuda_tensor_sin(tensor_t *tensor);
tensor_t *cuda_tensor_cos(tensor_t *tensor);
tensor_t *cuda_tensor_power_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_add_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_subtract_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_multiply_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_divide_scalar(tensor_t *a, float scalar);

tensor_t *cuda_tensor_greater(tensor_t *a, tensor_t *b);
tensor_t *cuda_tensor_less(tensor_t *a, tensor_t *b);
tensor_t *cuda_tensor_equal(tensor_t *a, tensor_t *b);
tensor_t *cuda_tensor_not_equal(tensor_t *a, tensor_t *b);
tensor_t *cuda_tensor_greater_equal(tensor_t *a, tensor_t *b);
tensor_t *cuda_tensor_less_equal(tensor_t *a, tensor_t *b);

tensor_t *cuda_tensor_greater_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_less_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_equal_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_not_equal_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_greater_equal_scalar(tensor_t *a, float scalar);
tensor_t *cuda_tensor_less_equal_scalar(tensor_t *a, float scalar);

tensor_t* perform_broadcast_operation(tensor_t *a, tensor_t *b, int operation_type);

int* cuda_tensor_get_shape(tensor_t* tensor);
tensor_t *cuda_tensor_transpose(tensor_t *tensor);

tensor_t *cuda_tensor_copy(tensor_t *tensor);
size_t cuda_tensor_size(tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif