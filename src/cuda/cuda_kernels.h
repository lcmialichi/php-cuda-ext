#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include "../tensor.h"

typedef struct ConcatParams
{
    void *input_ptrs[MAX_CONCAT_TENSORS];
    int input_axis_sizes[MAX_CONCAT_TENSORS];
    size_t input_axis_offsets[MAX_CONCAT_TENSORS];
    size_t input_strides_axis[MAX_CONCAT_TENSORS];
    int num_tensors;
    size_t outer_dims;
    size_t inner_dims;
    size_t output_stride;
    int output_axis_size;
} ConcatParams;

#ifdef __cplusplus
extern "C"
{
#endif

    void launch_fill_kernel(float *data, float value, size_t size);
    int launch_scale_kernel_host(float *data, size_t size, float min_value, float max_value);
    void launch_clip_kernel(float *a, float min_val, float max_val, float *result, int n);
    void launch_relu_kernel(float *a, float *result, int n);
    void launch_sigmoid_kernel(float *a, float *result, int n);
    void launch_tanh_kernel(float *a, float *result, int n);
    int cuda_matmul_launcher(float *a, float *b, float *c,
                             int m, int n, int k,
                             size_t a_stride0, size_t a_stride1,
                             size_t b_stride0, size_t b_stride1,
                             size_t c_stride0, size_t c_stride1);

    int cuda_batched_matmul_launcher(float *a, float *b, float *c,
                                     int *a_shape, size_t *a_strides,
                                     int *b_shape, size_t *b_strides,
                                     int *c_shape, size_t *c_strides,
                                     int a_ndims, int b_ndims);
    int launch_concat_kernel_host(
        tensor_t **input_tensors,
        int num_tensors,
        tensor_t *output_tensor,
        int axis,
        size_t *input_axis_offsets,
        size_t *input_strides_axis,
        size_t output_stride_axis,
        size_t outer_dims,
        size_t inner_dims,
        int output_axis_size);

#ifdef __cplusplus
}
#endif

#endif