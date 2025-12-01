#include <cuda_runtime.h>
#include "operation_functors.cuh"
#include "reduction_ops.cuh"
#include "reduction_ops.h"

extern "C"
{

#define DEFINE_REDUCTION_WRAPPER(name, Op)                                              \
    void name(float *input, float *output,                                              \
              int *input_shape, int input_ndims,                                        \
              int *result_shape,                                                        \
              size_t *input_strides,                                                    \
              int result_ndims, int axis,                                               \
              size_t total_elements_out, size_t input_base_offset)                      \
    {                                                                                   \
        launch_reduce_op<Op>(                                                           \
            input, output,                                                              \
            input_shape,                                                                \
            input_ndims,                                                                \
            result_shape,                                                               \
            input_strides, result_ndims, axis, total_elements_out, input_base_offset); \
    }

#define DEFINE_ARG_REDUCTION_WRAPPER(name, Op)                            \
    void name(float *input, int *output,                                  \
              int *input_shape, int input_ndims,                          \
              size_t *input_strides,                                      \
              int axis,                                                   \
              size_t total_elements_out, size_t input_base_offset)        \
    {                                                                     \
        launch_arg_reduce<Op>(                                            \
            input, output,                                                \
            input_shape,                                                  \
            input_ndims,                                                  \
            input_strides, axis, total_elements_out, input_base_offset); \
    }

    DEFINE_REDUCTION_WRAPPER(launch_reduce_sum, AddOp)
    DEFINE_REDUCTION_WRAPPER(launch_reduce_max, MaxOp)
    DEFINE_REDUCTION_WRAPPER(launch_reduce_min, MinOp)
    DEFINE_REDUCTION_WRAPPER(launch_reduce_prod, MulOp)

    DEFINE_ARG_REDUCTION_WRAPPER(launch_arg_min, ArgMinOp)
    DEFINE_ARG_REDUCTION_WRAPPER(launch_arg_max, ArgMaxOp)
}
