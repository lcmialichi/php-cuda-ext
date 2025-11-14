#ifndef TENSOR_H
#define TENSOR_H

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#define MAX_DIMS 10

typedef enum {
    SLICE_ALL = 0,
    SLICE_RANGE = 1, 
    SLICE_INDEX = 2
} slice_type_t;

typedef struct {
    slice_type_t type;
    union {
        struct {
            int start;
            int end;
        } range;
        int index;
    } data;
} slice_info_t;

typedef struct tensor
{
    float *data;
    int *shape;
    int ndims;
    size_t total_size;
    cudnnTensorDescriptor_t desc;
    int ref_count;
    
    int is_view;
    size_t gpu_offset;
    slice_info_t *slices;
    struct tensor *base_tensor;
    int num_slices;
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
    
    tensor_t *resolve_result_tensor(tensor_t *t);
    tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices);
    tensor_t *cuda_tensor_create(const int shape[], int ndims, const float data[]);
    tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims);
    tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, float value);
    tensor_t *cuda_tensor_create_empty(const int shape[], int ndims);

#ifdef __cplusplus
}
#endif

#endif