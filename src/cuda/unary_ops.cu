#include <cuda_runtime.h>
#include "unary_ops.h"
#include "unary_ops.cuh"
#include "operation_functors.cuh"

extern "C"
{
#define DEFINE_UNARY_WRAPPER(name, Op)                                                     \
    void name(float *base,                                                                 \
              float *result,                                                               \
              size_t base_offset,                                                          \
              int *shape,                                                                  \
              size_t *strides,                                                             \
              int ndims,                                                                   \
              size_t total_size)                                                           \
    {                                                                                      \
        launch_unary_op<Op>(base, result, base_offset, shape, strides, ndims, total_size); \
    }

    DEFINE_UNARY_WRAPPER(launch_unary_exp_kernel, ExpOp)
    DEFINE_UNARY_WRAPPER(launch_unary_sqrt_kernel, SqrtOp)
    DEFINE_UNARY_WRAPPER(launch_unary_log_kernel, LogOp)
    DEFINE_UNARY_WRAPPER(launch_unary_sin_kernel, SinOp)
    DEFINE_UNARY_WRAPPER(launch_unary_cos_kernel, CosOp)
    DEFINE_UNARY_WRAPPER(launch_unary_tan_kernel, TanOp)
    DEFINE_UNARY_WRAPPER(launch_unary_abs_kernel, AbsOp)
    DEFINE_UNARY_WRAPPER(launch_unary_neg_kernel, NegOp)
    DEFINE_UNARY_WRAPPER(launch_unary_floor_kernel, FloorOp)
    DEFINE_UNARY_WRAPPER(launch_unary_ceil_kernel, CeilOp)
    DEFINE_UNARY_WRAPPER(launch_unary_round_kernel, RoundOp)
}