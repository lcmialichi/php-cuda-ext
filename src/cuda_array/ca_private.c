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

unary_fn get_unary_fn(operation_type_t op)
{
    for (int i = 0; i < sizeof(unary_dispatch) / sizeof(UnaryDispatchEntry); i++)
        if (unary_dispatch[i].op == op)
            return unary_dispatch[i].fn;

    return NULL;
}

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);

    int result_shape[MAX_DIMS];
    int result_dims;
    int a_strides[MAX_DIMS] = {0};
    int b_strides[MAX_DIMS] = {0};
    size_t total_elements;

    if (!prepare_broadcast_operation(a, b, result_shape, &result_dims,
                                     a_strides, b_strides, &total_elements))
    {
        zend_throw_error(NULL, "Broadcast failed: shapes %s and %s are incompatible",
                         tensor_shape_as_string(a),
                         tensor_shape_as_string(b));
        return NULL;
    }

    if (can_safely_cast_to(a->dtype, b->dtype) == 0)
    {
        zend_throw_error(NULL, "Failed to promote type %s to %s",
                         dtype_to_string(a->dtype),
                         dtype_to_string(b->dtype));
    }

    dtype_t promoted_type = promote_types_for_arithmetic(a->dtype, b->dtype, operation_type);
    tensor_t *result = cuda_tensor_create_empty_dtype(result_shape, result_dims, promoted_type);
    if (!result)
    {
        zend_throw_error(NULL, "Failed to create result tensor for operation between %s%s and %s%s",
                         dtype_to_string(a->dtype),
                         tensor_shape_as_string(a),
                         dtype_to_string(b->dtype),
                         tensor_shape_as_string(b));
        return NULL;
    }

    if (a->data == NULL || b->data == NULL || result->data == NULL)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    launch_broadcast(a->data, a->dtype, b->data, b->dtype, result->data,
                     promoted_type, operation_type,
                     a_strides, a->ndims,
                     b_strides, b->ndims,
                     result_shape, result_dims,
                     total_elements, a->offset, b->offset);

    cudaError_t status = cudaDeviceSynchronize();
    return (status == cudaSuccess) ? result : NULL;
}

tensor_t *cuda_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);

    dtype_t promoted_type = promote_scalar_for_arithmetic(a->dtype, scalar.dtype, operation_type);
    tensor_t *result = cuda_tensor_create_empty_dtype(a->shape, a->ndims, promoted_type);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    launch_scalar(a->data,
                  a->dtype,
                  scalar,
                  result->data,
                  promoted_type,
                  operation_type,
                  a->offset,
                  a->d_shape,
                  a->d_strides,
                  a->ndims,
                  a->total_size,
                  is_contiguous(a));

    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_inv_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);

    dtype_t promoted_type = promote_scalar_for_arithmetic(a->dtype, scalar.dtype, operation_type);
    tensor_t *result = cuda_tensor_create_empty_dtype(a->shape, a->ndims, promoted_type);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    launch_scalar_inv(
        a->data,
        a->dtype,
        scalar,
        result->data,
        promoted_type,
        operation_type,
        a->offset,
        a->d_shape,
        a->d_strides,
        a->ndims,
        a->total_size,
        is_contiguous(a));

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

    launch_arg_reduction(
        input->data,
        result->data,
        input->dtype,
        operation_type,
        input->shape,
        input->ndims,
        result->shape,
        input->strides,
        result->ndims,
        axis,
        total_elements_out,
        input->offset);
        
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

    result = cuda_tensor_create_empty_dtype(result_shape_arr, result_ndims, input->dtype);
    if (!result)
        return NULL;

    if (operation_type == OP_REDUCE_MEAN)
    {
        size_t block_size = input->shape[axis];
        zend_throw_error(NULL, "OP_REDUCE_MEAN not implemented yet.");
    }

    launch_reduction(input->data, result->data, input->dtype, operation_type, input->shape, input->ndims,
                     result_shape_arr, input->strides, result_ndims, axis, total_elements_out, input->offset);

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
