#include "ca_private.h"
#include "broadcast_ops.h"
#include "reduction_ops.h"
#include "unary_ops.h"
#include "tensor_fabric.h"
#include "scalar_ops.h"
#include "operations.h"
#include <stdlib.h>
#include <string.h>
#include "php.h"
#include "tensor.h"

ScalarDispatchEntry scalar_dispatch[] = {
    {OP_ADD, launch_scalar_add_kernel},
    {OP_SUB, launch_scalar_subtract_kernel},
    {OP_MUL, launch_scalar_multiply_kernel},
    {OP_DIV, launch_scalar_divide_kernel},
    {OP_POW, launch_scalar_power_kernel},
    {OP_GT, launch_scalar_greater_kernel},
    {OP_LT, launch_scalar_less_kernel},
    {OP_EQ, launch_scalar_equal_kernel},
    {OP_NE, launch_scalar_not_equal_kernel},
    {OP_GE, launch_scalar_greater_equal_kernel},
    {OP_LE, launch_scalar_less_equal_kernel},
};

ScalarDispatchEntry inv_scalar_dispatch[] = {
    {OP_ADD, launch_inv_scalar_add_kernel},
    {OP_SUB, launch_inv_scalar_subtract_kernel},
    {OP_MUL, launch_inv_scalar_multiply_kernel},
    {OP_DIV, launch_inv_scalar_divide_kernel},
    {OP_POW, launch_inv_scalar_power_kernel},
    {OP_GT, launch_inv_scalar_greater_kernel},
    {OP_LT, launch_inv_scalar_less_kernel},
    {OP_EQ, launch_inv_scalar_equal_kernel},
    {OP_NE, launch_inv_scalar_not_equal_kernel},
    {OP_GE, launch_inv_scalar_greater_equal_kernel},
    {OP_LE, launch_inv_scalar_less_equal_kernel},
};

BroadcastDispatchEntry broadcast_dispatch[] = {
    {OP_ADD, launch_broadcast_add},
    {OP_SUB, launch_broadcast_subtract},
    {OP_MUL, launch_broadcast_multiply},
    {OP_DIV, launch_broadcast_divide},
    {OP_POW, launch_broadcast_power},
    {OP_GT, launch_broadcast_greater},
    {OP_LT, launch_broadcast_less},
    {OP_EQ, launch_broadcast_equal},
    {OP_NE, launch_broadcast_not_equal},
    {OP_GE, launch_broadcast_greater_equal},
    {OP_LE, launch_broadcast_less_equal}};

UnaryDispatchEntry unary_dispatch[] = {
    {OP_EXP, launch_unary_exp_kernel},
    {OP_SQRT, launch_unary_sqrt_kernel},
    {OP_LOG, launch_unary_log_kernel},
    {OP_SIN, launch_unary_sin_kernel},
    {OP_COS, launch_unary_cos_kernel},
    {OP_TAN, launch_unary_tan_kernel},
    {OP_ABS, launch_unary_abs_kernel},
    {OP_FLOOR, launch_unary_floor_kernel},
    {OP_CEIL, launch_unary_ceil_kernel},
    {OP_ROUND, launch_unary_round_kernel},
    {OP_NEG, launch_unary_neg_kernel}};

ReductionDispatchEntry reduction_dispatch[] = {
    {OP_REDUCE_SUM, launch_reduce_sum},
    {OP_REDUCE_MAX, launch_reduce_max},
    {OP_REDUCE_MIN, launch_reduce_min},
    {OP_REDUCE_PROD, launch_reduce_prod},
};

ReductionArgDispatchEntry reduction_arg_dispatch[] = {
    {OP_ARG_MAX, launch_arg_max},
    {OP_ARG_MIN, launch_arg_min},
};

scalar_fn get_scalar_fn(operation_type_t op)
{
    for (int i = 0; i < sizeof(scalar_dispatch) / sizeof(ScalarDispatchEntry); i++)
        if (scalar_dispatch[i].op == op)
            return scalar_dispatch[i].fn;

    return NULL;
}

scalar_fn get_inv_scalar_fn(operation_type_t op)
{
    for (int i = 0; i < sizeof(inv_scalar_dispatch) / sizeof(ScalarDispatchEntry); i++)
        if (inv_scalar_dispatch[i].op == op)
            return inv_scalar_dispatch[i].fn;

    return NULL;
}

unary_fn get_unary_fn(operation_type_t op)
{
    for (int i = 0; i < sizeof(unary_dispatch) / sizeof(UnaryDispatchEntry); i++)
        if (unary_dispatch[i].op == op)
            return unary_dispatch[i].fn;

    return NULL;
}

broadcast_fn get_broadcast_fn(operation_type_t op)
{
    for (int i = 0; i < sizeof(broadcast_dispatch) / sizeof(BroadcastDispatchEntry); i++)
        if (broadcast_dispatch[i].op == op)
            return broadcast_dispatch[i].fn;

    return NULL;
}

reduction_fn get_reduction_fn(operation_type_t op)
{
    for (int i = 0; i < sizeof(reduction_dispatch) / sizeof(ReductionDispatchEntry); i++)
        if (reduction_dispatch[i].op == op)
            return reduction_dispatch[i].fn;

    return NULL;
}

reduction_arg_fn get_reduction_arg_fn(operation_type_t op)
{
    for (int i = 0; i < sizeof(reduction_arg_dispatch) / sizeof(ReductionArgDispatchEntry); i++)
        if (reduction_arg_dispatch[i].op == op)
            return reduction_arg_dispatch[i].fn;

    return NULL;
}

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    broadcast_fn func = get_broadcast_fn(operation_type);

    int result_shape[MAX_DIMS];
    int result_dims;
    int a_strides[MAX_DIMS] = {0};
    int b_strides[MAX_DIMS] = {0};
    size_t total_elements;

    if (!prepare_broadcast_operation(a, b, result_shape, &result_dims,
                                     a_strides, b_strides, &total_elements))
    {
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(result_shape, result_dims);

    if (func == NULL)
    {
        php_error_docref(NULL, E_ERROR, "Operation handler not found.");
        return NULL;
    }

    if (!result)
    {
        zend_throw_error(NULL, "Broadcast failed: shapes %s and %s are incompatible",
                         tensor_shape_as_string(a),
                         tensor_shape_as_string(b));
        return NULL;
    }

    if (a->data == NULL || b->data == NULL || result->data == NULL)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    func(a->data, b->data, result->data,
         a_strides, a->ndims,
         b_strides, b->ndims,
         result_shape, result_dims,
         total_elements, a->offset, b->offset);

    cudaError_t status = cudaDeviceSynchronize();
    return (status == cudaSuccess) ? result : NULL;
}

tensor_t *cuda_scalar_op(tensor_t *a, float scalar, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    tensor_t *result = resolve_result_tensor(a);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    scalar_fn func = get_scalar_fn(operation_type);
    if (func == NULL)
    {
        php_error_docref(NULL, E_ERROR, "Operation handler not found.");
        return NULL;
    }

    func(a->data, scalar, result->data, a->offset, a->shape, a->strides, a->ndims, a->total_size);
    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_inv_scalar_op(tensor_t *a, float scalar, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    tensor_t *result = resolve_result_tensor(a);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    scalar_fn func = get_inv_scalar_fn(operation_type);
    if (func == NULL)
    {
        php_error_docref(NULL, E_ERROR, "Operation handler not found.");
        return NULL;
    }

    func(a->data, scalar, result->data, a->offset, a->shape, a->strides, a->ndims, a->total_size);
    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_unary_op(tensor_t *a, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);

    unary_fn func = get_unary_fn(operation_type);
    if (func == NULL)
    {
        php_error_docref(NULL, E_ERROR, "Operation handler not found.");
        return NULL;
    }
    tensor_t *result = resolve_result_tensor(a);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    func(a->data, result->data, a->offset, a->d_shape, a->d_strides, a->ndims, a->total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Square root operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reduce_arg(tensor_t *input, int axis, operation_type_t operation_type)
{
    int result_shape_arr[MAX_DIMS];
    size_t total_elements_out;

    int result_ndims = calculate_reduction_shape(input, axis, result_shape_arr, &total_elements_out);
    if (total_elements_out == 0 && result_ndims > 0)
    {
        return NULL;
    }

    tensor_t *result = NULL;
    cudaError_t err = cudaSuccess;
    result = cuda_tensor_create_int(result_shape_arr, result_ndims, NULL);
    if (!result)
        return NULL;

    reduction_arg_fn func = get_reduction_arg_fn(operation_type);
    func(input->data, result->data, input->shape, input->ndims, input->strides, axis, total_elements_out, input->offset);
    if (!result)
    {
        zend_throw_error(NULL, "CudaArray creation failed during reduction.");
        return NULL;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        zend_throw_error(NULL, "Failed to synchronize device after reduction: %s", cudaGetErrorString(err));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reduce(tensor_t *input, int axis, operation_type_t operation_type)
{
    int result_shape_arr[MAX_DIMS];
    size_t total_elements_out;

    int result_ndims = calculate_reduction_shape(input, axis, result_shape_arr, &total_elements_out);
    if (total_elements_out == 0 && result_ndims > 0)
    {
        return NULL;
    }

    tensor_t *result = NULL;
    cudaError_t err = cudaSuccess;

    reduction_fn func = get_reduction_fn(operation_type);
    if (func == NULL)
    {
        zend_throw_error(NULL, "Reduction operation handler not found for type %d.", operation_type);
        return NULL;
    }

    result = cuda_tensor_create_empty(result_shape_arr, result_ndims);
    if (!result)
        return NULL;

    func(input->data, result->data, input->shape, input->ndims,
         result_shape_arr, input->strides, result_ndims, axis, total_elements_out, input->offset);

    if (operation_type == OP_REDUCE_MEAN)
    {
        size_t block_size = input->shape[axis];
        zend_throw_error(NULL, "OP_REDUCE_MEAN not implemented yet.");
    }

    if (!result)
    {
        zend_throw_error(NULL, "CudaArray creation failed during reduction.");
        return NULL;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        zend_throw_error(NULL, "Failed to synchronize device after reduction: %s", cudaGetErrorString(err));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reshape(tensor_t *original, int *new_shape, int new_ndims)
{
    if (original == NULL || new_shape == NULL || new_ndims <= 0)
    {
        return NULL;
    }

    size_t original_size = 1;
    for (int i = 0; i < original->ndims; i++)
    {
        original_size *= original->shape[i];
    }

    size_t new_size_known = 1;
    int wildcard_index = -1;
    int final_shape[MAX_DIMS];

    for (int i = 0; i < new_ndims; i++)
    {
        final_shape[i] = new_shape[i];

        if (new_shape[i] < 0)
        {
            if (wildcard_index != -1)
            {
                php_error_docref(NULL, E_WARNING, "Reshape allows only one wildcard dimension (-1) in the new shape.");
                return NULL;
            }
            wildcard_index = i;
        }
        else if (new_shape[i] == 0)
        {
            php_error_docref(NULL, E_WARNING, "Reshape dimension cannot be 0.");
            return NULL;
        }
        else
        {
            new_size_known *= new_shape[i];
        }
    }

    if (wildcard_index != -1)
    {
        if (original_size % new_size_known != 0)
        {
            php_error_docref(NULL, E_WARNING, "Cannot reshape array of size %zu into shape with known elements %zu.", original_size, new_size_known);
            return NULL;
        }
        final_shape[wildcard_index] = original_size / new_size_known;
        new_size_known = original_size;
    }

    if (original_size != new_size_known)
    {
        php_error_docref(NULL, E_WARNING, "Reshape requires that the number of elements remains the same. Original: %zu, New: %zu.", original_size, new_size_known);
        return NULL;
    }

    if (!is_contiguous(original))
    {
        php_error_docref(NULL, E_WARNING, "Reshape of non-contiguous tensor requires a memory copy/reorder operation, which is not yet implemented. Returning NULL.");
        return NULL;
    }

    size_t new_strides[MAX_DIMS];
    new_strides[new_ndims - 1] = 1;
    for (int i = new_ndims - 2; i >= 0; i--)
    {
        new_strides[i] = new_strides[i + 1] * final_shape[i + 1];
    }

    tensor_t *reshaped = cuda_tensor_create_view(
        original,
        final_shape,
        new_strides,
        new_ndims,
        original->offset,
        original->total_size);

    return reshaped;
}

tensor_t *cuda_tensor_transpose(tensor_t *tensor, int *axis, int axis_len)
{
    if (tensor == NULL || axis == NULL)
    {
        return NULL;
    }

    if (axis_len != tensor->ndims)
    {
        return NULL;
    }

    for (int i = 0; i < axis_len; i++)
    {
        if (axis[i] < 0 || axis[i] >= tensor->ndims)
        {
            return NULL;
        }
    }

    int new_shape[MAX_DIMS];
    size_t new_strides[MAX_DIMS];
    int ndims = tensor->ndims;

    for (int i = 0; i < ndims; i++)
    {
        new_shape[i] = tensor->shape[axis[i]];
        new_strides[i] = tensor->strides[axis[i]];
    }

    tensor_t *transposed = cuda_tensor_create_view(
        tensor,
        new_shape,
        new_strides,
        tensor->ndims,
        tensor->offset,
        tensor->total_size);

    return transposed ? transposed : NULL;
}

tensor_t *cuda_tensor_matmul_nd(tensor_t *a, tensor_t *b)
{
    CUDA_CHECK_AND_RETURN_NULL(a);

    int result_ndims;
    int result_shape[MAX_DIMS];
    if (prepare_matmul_result_shape(
            a->ndims,
            a->shape,
            b->ndims,
            b->shape,
            &result_ndims,
            result_shape) == 0)
    {
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(result_shape, result_ndims);
    if (result == NULL)
    {
        return NULL;
    }

    int status = cuda_batched_matmul_nd_launcher(
        a->data, b->data, result->data,
        a->d_shape, a->d_strides, a->ndims,
        b->d_shape, b->d_strides, b->ndims,
        result->d_shape, result->d_strides, result->ndims);

    if (status == 0 || !result->data)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (a->ndims < 2 || b->ndims < 2)
    {
        return NULL;
    }

    LAZY_COPY_METADATA(a);
    LAZY_COPY_METADATA(b);

    if (a->ndims != 2 || b->ndims != 2)
    {
        return cuda_tensor_matmul_nd(a, b);
    }

    if (a->shape[1] != b->shape[0])
    {
        return NULL;
    }

    int result_shape[2] = {a->shape[0], b->shape[1]};

    tensor_t *result = cuda_tensor_create_empty(result_shape, 2);
    if (result == NULL)
    {
        return NULL;
    }

    int status = cuda_matmul_launcher(
        a->data, b->data, result->data,
        a->shape[0], a->shape[1], b->shape[1],
        a->strides[0], a->strides[1],
        b->strides[0], b->strides[1],
        result->strides[0], result->strides[1]);

    if (status == 0)
    {
        efree(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_copy(tensor_t *tensor)
{
    if (!tensor)
        return NULL;

    tensor_t *copy = cuda_tensor_create_empty(tensor->shape, tensor->ndims);
    if (!copy)
        return NULL;

    cudaError_t cuda_status = cudaMemcpy(
        copy->data, tensor->data,
        tensor->total_size * sizeof(float),
        cudaMemcpyDeviceToDevice);

    if (cuda_status != cudaSuccess)
    {
        cuda_tensor_destroy(copy);
        return NULL;
    }

    return copy;
}
