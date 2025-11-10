#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C"
{
#endif

    void launch_add_kernel(float *a, float *b, float *result, int n);
    void launch_subtract_kernel(float *a, float *b, float *result, int n);
    void launch_multiply_kernel(float *a, float *b, float *result, int n);
    void launch_divide_kernel(float *a, float *b, float *result, int n);
    void launch_power_kernel(float *a, float *b, float *result, int n);

    void launch_scalar_add_kernel(float *a, float scalar, float *result, int n);
    void launch_scalar_subtract_kernel(float *a, float scalar, float *result, int n);
    void launch_scalar_multiply_kernel(float *a, float scalar, float *result, int n);
    void launch_scalar_divide_kernel(float *a, float scalar, float *result, int n);
    void launch_scalar_power_kernel(float *a, float scalar, float *result, int n);

    void launch_sqrt_kernel(float *a, float *result, int n);
    void launch_exp_kernel(float *a, float *result, int n);
    void launch_log_kernel(float *a, float *result, int n);
    void launch_sin_kernel(float *a, float *result, int n);
    void launch_cos_kernel(float *a, float *result, int n);
    void launch_tan_kernel(float *a, float *result, int n);
    void launch_abs_kernel(float *a, float *result, int n);
    void launch_negate_kernel(float *a, float *result, int n);
    void launch_reciprocal_kernel(float *a, float *result, int n);

    void launch_greater_kernel(float *a, float *b, float *result, int n);
    void launch_less_kernel(float *a, float *b, float *result, int n);
    void launch_equal_kernel(float *a, float *b, float *result, int n);
    void launch_greater_equal_kernel(float *a, float *b, float *result, int n);
    void launch_less_equal_kernel(float *a, float *b, float *result, int n);
    void launch_not_equal_kernel(float *a, float *b, float *result, int n);

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