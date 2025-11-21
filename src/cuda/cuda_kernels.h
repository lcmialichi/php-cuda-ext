#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C"
{
#endif

    void launch_fill_kernel(float *data, float value, size_t size);
    void launch_sum_kernel(float *a, float *result, int n);
    void launch_max_kernel(float *a, float *result, int n);
    void launch_min_kernel(float *a, float *result, int n);

    void launch_clip_kernel(float *a, float min_val, float max_val, float *result, int n);
    void launch_relu_kernel(float *a, float *result, int n);
    void launch_sigmoid_kernel(float *a, float *result, int n);
    void launch_tanh_kernel(float *a, float *result, int n);
    void launch_scatter(float *dest_data,
                        const float *src_data,
                        const int *indices,
                        size_t num_indices,
                        size_t slice_size);

#ifdef __cplusplus
}
#endif

#endif