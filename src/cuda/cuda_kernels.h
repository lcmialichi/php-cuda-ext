#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C"
{
#endif
    void launch_sqrt_kernel(float *a, float *result, int n);
    void launch_exp_kernel(float *a, float *result, int n);
    void launch_log_kernel(float *a, float *result, int n);
    void launch_sin_kernel(float *a, float *result, int n);
    void launch_cos_kernel(float *a, float *result, int n);
    void launch_tan_kernel(float *a, float *result, int n);
    void launch_abs_kernel(float *a, float *result, int n);
    void launch_negate_kernel(float *a, float *result, int n);
    void launch_reciprocal_kernel(float *a, float *result, int n);

    void launch_sum_kernel(float *a, float *result, int n);
    void launch_max_kernel(float *a, float *result, int n);
    void launch_min_kernel(float *a, float *result, int n);

    void launch_clip_kernel(float *a, float min_val, float max_val, float *result, int n);
    void launch_relu_kernel(float *a, float *result, int n);
    void launch_sigmoid_kernel(float *a, float *result, int n);
    void launch_tanh_kernel(float *a, float *result, int n);

#ifdef __cplusplus
}
#endif

#endif