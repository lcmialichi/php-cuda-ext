#ifndef TENSOR_H
#define TENSOR_H

#include <cuda_runtime.h>

#define MAX_DIMS 10
#define MAX_CONCAT_TENSORS 10
#define DTYPE_FLOAT 1
#define DTYPE_INT 2

struct _operation_t;

typedef enum
{
    SLICE_ALL = 0,
    SLICE_RANGE = 1,
    SLICE_INDEX = 2
} slice_type_t;

typedef struct
{
    slice_type_t type;
    union
    {
        struct
        {
            int start;
            int end;
        } range;
        int index;
    } data;
} slice_info_t;

typedef enum _tensor_type_t
{
    TENSOR_TYPE_INPUT,
    TENSOR_TYPE_OUTPUT,
    TENSOR_TYPE_TEMP
} tensor_type_t;

typedef struct tensor
{
    void *data;
    int dtype;
    void *device_ptr;
    int *shape;
    size_t element_size;
    size_t *strides;
    int ndims;
    size_t total_size;
    int ref_count;
    size_t allocated_size;
    int is_view;
    size_t gpu_offset;
    slice_info_t *slices;
    struct tensor *base_tensor;
    int num_slices;
    size_t *d_strides;
    int *d_shape;
    int is_on_gpu;
    int is_dirty;
} tensor_t;

#ifdef __cplusplus
extern "C"
{
#endif

#define CUDA_CHECK_AND_RETURN_NULL(__tensor__)                                           \
    do                                                                                   \
    {                                                                                    \
        if (UNEXPECTED(!cuda_initialized() || (__tensor__) == NULL))                     \
        {                                                                                \
            php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL"); \
            return NULL;                                                                 \
        }                                                                                \
    } while (0)

#define CUDA_CHECK_AND_RETURN_FAILURE(__tensor__)                                        \
    do                                                                                   \
    {                                                                                    \
        if (UNEXPECTED(!cuda_initialized() || (__tensor__) == NULL))                     \
        {                                                                                \
            php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL"); \
            return 0;                                                                    \
        }                                                                                \
    } while (0)

#define LAZY_COPY_METADATA(__tensor__)                                    \
    do                                                                    \
    {                                                                     \
        if (__tensor__->d_shape == NULL || __tensor__->d_strides == NULL) \
        {                                                                 \
            lazy_copy_metadata_to_gpu(__tensor__);                        \
        }                                                                 \
    } while (0)

    int tensor_init();
    int cuda_initialized();
    void cuda_set_initialized(int status);
    void cuda_tensor_destroy(tensor_t *tensor);
    int is_contiguous(tensor_t *tensor);
    void lazy_copy_metadata_to_gpu(tensor_t *t);
    char *tensor_shape_as_string(tensor_t *tensor);

    tensor_t *cuda_tensor_create_sliced_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices);
    tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, int *shape, size_t *strides, int dims, size_t offset, size_t total_size);
    tensor_t *cuda_tensor_create_dim_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices);

    int cuda_tensor_set_scalar(tensor_t *tensor, size_t element_offset, float scalar_value);
    int cuda_tensor_set_tensor(tensor_t *base_tensor, size_t element_offset, tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif