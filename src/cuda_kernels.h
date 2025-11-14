#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#define OP_ADD 0
#define OP_SUB 1
#define OP_MUL 2
#define OP_DIV 3
#define OP_POW 4
#define OP_GT 5
#define OP_LT 6
#define OP_EQ 7
#define OP_NE 8
#define OP_GE 9
#define OP_LE 10

#ifdef __cplusplus
extern "C"
{
#endif

    void launch_broadcast_kernel(float *a, float *b, float *result,
                                 int *a_strides, int a_dims,
                                 int *b_strides, int b_dims,
                                 int *result_shape, int result_dims,
                                 size_t total_elements, int operation_type);


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