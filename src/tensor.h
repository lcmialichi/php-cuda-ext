#ifndef TENSOR_H
#define TENSOR_H

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

typedef struct
{
    float *data;
    int *shape;
    int ndims;
    size_t total_size;
    cudnnTensorDescriptor_t desc;
    int ref_count;
} tensor_t;

static cudnnHandle_t cudnn_handle = NULL;
static cublasHandle_t cublas_handle = NULL;

#ifdef __cplusplus
extern "C"
{
#endif

    int tensor_init();
    int cuda_initialized();
    void cuda_set_initialized(int status);
    void cuda_tensor_destroy(tensor_t *tensor);
    tensor_t *cuda_tensor_create(const int shape[], int ndims, const float data[]);
    tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims);
    tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, float value);
    tensor_t *cuda_tensor_create_empty(const int shape[], int ndims);

#ifdef __cplusplus
}
#endif

#endif