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

static char *tensor_shape_as_string(tensor_t *tensor);

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

int calculate_broadcast_shape(int *a_shape, int a_dims, int *b_shape, int b_dims, int *result_shape, int *result_dims)
{
    *result_dims = (a_dims > b_dims) ? a_dims : b_dims;

    int offset_a = *result_dims - a_dims;
    int offset_b = *result_dims - b_dims;

    for (int i = 0; i < *result_dims; i++)
    {
        int a_dim = (i < offset_a) ? 1 : a_shape[i - offset_a];
        int b_dim = (i < offset_b) ? 1 : b_shape[i - offset_b];

        if (a_dim == b_dim)
        {
            result_shape[i] = a_dim;
        }
        else if (a_dim == 1)
        {
            result_shape[i] = b_dim;
        }
        else if (b_dim == 1)
        {
            result_shape[i] = a_dim;
        }
        else
        {
            return 0;
        }
    }

    return 1;
}

int calculate_broadcast_stride(int *result_shape, int result_dims, int dim_idx)
{
    int stride = 1;
    for (int i = result_dims - 1; i > dim_idx; i--)
    {
        stride *= result_shape[i];
    }
    return stride;
}

static void calculate_tensor_strides(tensor_t *tensor,
                                     int *result_shape,
                                     int result_dims,
                                     int *tensor_strides)
{
    long internal_stride = 1;
    for (int i = tensor->ndims - 1; i >= 0; i--)
    {
        if (tensor->is_view)
        {
            tensor_strides[i] = (int)tensor->strides[i];
        }
        else if (tensor->shape[i] == 1)
        {
            tensor_strides[i] = 0;
        }
        else
        {
            tensor_strides[i] = (int)internal_stride;
            internal_stride *= tensor->shape[i];
        }
    }
}

int prepare_broadcast_operation(tensor_t *a, tensor_t *b,
                                int *result_shape, int *result_dims,
                                int *a_strides, int *b_strides,
                                size_t *total_elements)
{
    if (!calculate_broadcast_shape(a->shape, a->ndims, b->shape, b->ndims,
                                   result_shape, result_dims))
    {
        return 0;
    }

    *total_elements = 1;
    for (int i = 0; i < *result_dims; i++)
    {
        *total_elements *= result_shape[i];
    }

    calculate_tensor_strides(a, result_shape, *result_dims, a_strides);
    calculate_tensor_strides(b, result_shape, *result_dims, b_strides);

    return 1;
}

static int calculate_reduction_shape(tensor_t *input, int axis, int *result_shape, size_t *total_elements_out_ptr)
{
    if (axis < 0 || axis >= input->ndims)
    {
        zend_throw_error(NULL, "Invalid axis %d for reduction operation. Must be between 0 and %d.", axis, input->ndims - 1);
        return 0;
    }

    size_t total_elements = 1;
    int j = 0;
    for (int i = 0; i < input->ndims; i++)
    {
        if (i != axis)
        {
            result_shape[j++] = input->shape[i];
            total_elements *= input->shape[i];
        }
    }

    *total_elements_out_ptr = total_elements;

    if (j == 0)
    {
        j = 1;
        result_shape[0] = 1;
        *total_elements_out_ptr = 1;
    }

    return j;
}

scalar_fn get_scalar_fn(int op)
{
    for (int i = 0; i < sizeof(scalar_dispatch) / sizeof(ScalarDispatchEntry); i++)
        if (scalar_dispatch[i].op == op)
            return scalar_dispatch[i].fn;

    return NULL;
}

unary_fn get_unary_fn(int op)
{
    for (int i = 0; i < sizeof(unary_dispatch) / sizeof(UnaryDispatchEntry); i++)
        if (unary_dispatch[i].op == op)
            return unary_dispatch[i].fn;

    return NULL;
}

broadcast_fn get_broadcast_fn(int op)
{
    for (int i = 0; i < sizeof(broadcast_dispatch) / sizeof(BroadcastDispatchEntry); i++)
        if (broadcast_dispatch[i].op == op)
            return broadcast_dispatch[i].fn;

    return NULL;
}

reduction_fn get_reduction_fn(int op)
{
    for (int i = 0; i < sizeof(reduction_dispatch) / sizeof(ReductionDispatchEntry); i++)
        if (reduction_dispatch[i].op == op)
            return reduction_dispatch[i].fn;

    return NULL;
}

reduction_arg_fn get_reduction_arg_fn(int op)
{
    for (int i = 0; i < sizeof(reduction_arg_dispatch) / sizeof(ReductionArgDispatchEntry); i++)
        if (reduction_arg_dispatch[i].op == op)
            return reduction_arg_dispatch[i].fn;

    return NULL;
}

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, int operation_type)
{
    if (!cuda_initialized())
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

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
         total_elements, a->gpu_offset, b->gpu_offset);

    cudaError_t status = cudaDeviceSynchronize();
    return (status == cudaSuccess) ? result : NULL;
}

tensor_t *cuda_scalar_op(tensor_t *a, float scalar, int operation_type)
{
    if (!cuda_initialized())
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

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

    func(a->data, scalar, result->data, a->gpu_offset, a->shape, a->strides, a->ndims, a->total_size);
    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_unary_op(tensor_t *a, int operation_type)
{
    if (!cuda_initialized() || a == NULL)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL");
        return NULL;
    }

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

    func(a->data, result->data, a->gpu_offset, a->d_shape, a->d_strides, a->ndims, a->total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Square root operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reduce_arg(tensor_t *input, int axis, int operation_type)
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
    func(input->data, result->data, input->shape, input->ndims, input->strides, axis, total_elements_out, input->gpu_offset);
    if (!result)
    {
        zend_throw_error(NULL, "Tensor creation failed during reduction.");
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

tensor_t *cuda_tensor_reduce(tensor_t *input, int axis, int operation_type)
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
         result_shape_arr, input->strides, result_ndims, axis, total_elements_out, input->gpu_offset);

    if (operation_type == OP_REDUCE_MEAN)
    {
        size_t block_size = input->shape[axis];
        zend_throw_error(NULL, "OP_REDUCE_MEAN not implemented yet.");
    }

    if (!result)
    {
        zend_throw_error(NULL, "Tensor creation failed during reduction.");
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
        original->gpu_offset,
        original->total_size);

    return reshaped;
}

tensor_t *cuda_tensor_transpose(tensor_t *tensor, int *axis, int axis_len)
{
    if (!cuda_initialized() || tensor == NULL || axis == NULL)
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
        tensor->gpu_offset,
        tensor->total_size);

    return transposed ? transposed : NULL;
}

tensor_t *cuda_tensor_matmul_nd(tensor_t *a, tensor_t *b)
{
    if (!cuda_initialized() || a == NULL || b == NULL)
    {
        return NULL;
    }

    int max_ndims = (a->ndims > b->ndims) ? a->ndims : b->ndims;
    int a_inner = a->shape[a->ndims - 1];
    int b_inner = b->shape[b->ndims - 2];

    if (a_inner != b_inner)
    {
        return NULL;
    }

    int result_ndims = max_ndims;
    int result_shape[MAX_DIMS];

    for (int i = 0; i < max_ndims - 2; i++)
    {
        int a_idx = a->ndims - max_ndims + i;
        int b_idx = b->ndims - max_ndims + i;

        int a_dim = (a_idx < 0) ? 1 : a->shape[a_idx];
        int b_dim = (b_idx < 0) ? 1 : b->shape[b_idx];

        if (a_dim == b_dim)
        {
            result_shape[i] = a_dim;
        }
        else if (a_dim == 1)
        {
            result_shape[i] = b_dim;
        }
        else if (b_dim == 1)
        {
            result_shape[i] = a_dim;
        }
        else
        {
            return NULL;
        }
    }

    result_shape[max_ndims - 2] = a->shape[a->ndims - 2];
    result_shape[max_ndims - 1] = b->shape[b->ndims - 1];

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
        efree(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b)
{
    if (!cuda_initialized() || a == NULL || b == NULL)
    {
        return NULL;
    }

    if (a->ndims != 2 || b->ndims != 2)
    {
        return cuda_tensor_matmul_nd(a, b);
    }

    if (a->ndims < 2 || b->ndims < 2)
    {
        return NULL;
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

static char *tensor_shape_as_string(tensor_t *tensor)
{
    if (tensor->ndims == 0)
    {
        char *result = (char *)emalloc(8);
        strcpy(result, "scalar");
        return result;
    }

    int buffer_size = tensor->ndims * 12 + 2;
    char *result = (char *)emalloc(buffer_size);

    char *ptr = result;
    *ptr++ = '(';

    for (int i = 0; i < tensor->ndims; i++)
    {
        if (i > 0)
        {
            *ptr++ = ',';
            *ptr++ = ' ';
        }
        ptr += sprintf(ptr, "%d", tensor->shape[i]);
    }

    *ptr++ = ')';
    *ptr = '\0';

    return result;
}