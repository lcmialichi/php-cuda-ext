#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

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

#ifdef __cplusplus
}
#endif

#endif