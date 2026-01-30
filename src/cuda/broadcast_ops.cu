#include <cuda_runtime.h>
#include "operation_functors.cuh"
#include "new_ops_func.cuh"
#include "broadcast_ops.cuh"
#include "broadcast_ops.h"

#define DISPATCH_DTYPE(DTYPE, ...) \
    switch (DTYPE)                 \
    {                              \
    case DTYPE_FLOAT32:            \
    {                              \
        typedef float scalar_t;    \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_FLOAT64:            \
    {                              \
        typedef double scalar_t;   \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT8:               \
    {                              \
        typedef int8_t scalar_t;   \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT16:              \
    {                              \
        typedef int16_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT32:              \
    {                              \
        typedef int32_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_INT64:              \
    {                              \
        typedef int64_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT8:              \
    {                              \
        typedef uint8_t scalar_t;  \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT16:             \
    {                              \
        typedef uint16_t scalar_t; \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT32:             \
    {                              \
        typedef uint32_t scalar_t; \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_UINT64:             \
    {                              \
        typedef uint64_t scalar_t; \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    case DTYPE_BOOL:               \
    {                              \
        typedef bool scalar_t;     \
        __VA_ARGS__;               \
        break;                     \
    }                              \
    default:                       \
        return;               \
    }

#define DISPATCH_OP(OP_TYPE, ...)          \
    switch (OP_TYPE)                       \
    {                                      \
    case OP_ADD:                           \
    {                                      \
        typedef AddOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_SUB:                           \
    {                                      \
        typedef SubOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_MUL:                           \
    {                                      \
        typedef MulOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_DIV:                           \
    {                                      \
        typedef DivOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_POW:                           \
    {                                      \
        typedef PowOpT<scalar_t> bin_op_t; \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_GT:                            \
    {                                      \
        typedef GTOpT<scalar_t> bin_op_t;   \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_LT:                            \
    {                                      \
        typedef LTOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_EQ:                            \
    {                                      \
        typedef EQOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_NE:                            \
    {                                      \
        typedef NEOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_GE:                            \
    {                                      \
        typedef GEOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    case OP_LE:                            \
    {                                      \
        typedef LEOpT<scalar_t> bin_op_t;  \
        __VA_ARGS__;                       \
        break;                             \
    }                                      \
    default:                               \
        return;                       \
    }

extern "C" void launch_broadcast(
    void *a, void *b, void *result,
    dtype_t dtype, operation_type_t op_type,
    int *a_strides, int a_dims,
    int *b_strides, int b_dims,
    int *result_shape, int result_dims,
    size_t total_elements, size_t a_offset, size_t b_offset)
{
    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP(op_type, {
            launch_broadcast_kernel<scalar_t, bin_op_t>(
                (scalar_t *)a,
                (scalar_t *)b,
                (scalar_t *)result,
                a_strides, a_dims,
                b_strides, b_dims,
                result_shape, result_dims,
                total_elements,
                a_offset,
                b_offset);
        });
    });
}

extern "C"
{

#define DEFINE_BROADCAST_WRAPPER(name, Op)                 \
    void name(float *a, float *b, float *result,           \
              int *a_strides, int a_dims,                  \
              int *b_strides, int b_dims,                  \
              int *result_shape, int result_dims,          \
              size_t total_elements,                       \
              size_t a_base_offset,                        \
              size_t b_base_offset)                        \
    {                                                      \
        launch_broadcast_op<Op>(                           \
            a, b, result,                                  \
            a_strides, a_dims,                             \
            b_strides, b_dims,                             \
            result_shape, result_dims,                     \
            total_elements, a_base_offset, b_base_offset); \
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
