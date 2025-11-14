#include <cuda_runtime.h>
#include "scalar_ops.h"
#include "scalar_ops.cuh"
#include "operation_functors.cuh"

extern "C"
{
#define DEFINE_SCALAR_WRAPPER(name, Op)                            \
    void name(float *a, float scalar, float *result, size_t total) \
    {                                                              \
        launch_scalar_op<Op>(a, scalar, result, total);            \
    }

    DEFINE_SCALAR_WRAPPER(launch_scalar_add_kernel, AddOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_subtract_kernel, SubOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_multiply_kernel, MulOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_divide_kernel, DivOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_power_kernel, PowOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_greater_kernel, GreaterOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_less_kernel, LessOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_equal_kernel, EqualOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_not_equal_kernel, NotEqualOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_greater_equal_kernel, GreaterEqualOp)
    DEFINE_SCALAR_WRAPPER(launch_scalar_less_equal_kernel, LessEqualOp)
}