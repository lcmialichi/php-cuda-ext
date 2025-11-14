#include <cuda_runtime.h>
#include "operation_functors.cuh"
#include "broadcast_ops.cuh"
#include "broadcast_ops.h"
#include "operations.h"

extern "C"
{

#define DEFINE_BROADCAST_WRAPPER(name, Op)        \
    void name(float *a, float *b, float *result,  \
              int *a_strides, int a_dims,         \
              int *b_strides, int b_dims,         \
              int *result_shape, int result_dims, \
              size_t total_elements)              \
    {                                             \
        launch_broadcast_op<Op>(                  \
            a, b, result,                         \
            a_strides, a_dims,                    \
            b_strides, b_dims,                    \
            result_shape, result_dims,            \
            total_elements);                      \
    }

    DEFINE_BROADCAST_WRAPPER(launch_broadcast_add, AddOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_subtract, SubOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_multiply, MulOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_divide, DivOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_power, PowOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_greater, GreaterOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_less, LessOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_equal, EqualOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_not_equal, NotEqualOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_greater_equal, GreaterEqualOp)
    DEFINE_BROADCAST_WRAPPER(launch_broadcast_less_equal, LessEqualOp)
}
