#include "ca_private.h"
#include "broadcast_ops.h"
#include "tensor_fabric.h"
#include "scalar_ops.h"
#include "unary_ops.h"
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
    for (int i = 0; i < tensor->ndims; i++)
    {
        if (tensor->is_view)
        {
            tensor_strides[i] = (int)tensor->strides[i];
            continue;
        }

        int result_dim_idx = result_dims - tensor->ndims + i;
        tensor_strides[i] = result_dim_idx >= 0 && tensor->shape[i] != 1
                                ? calculate_broadcast_stride(result_shape, result_dims, result_dim_idx)
                                : 0;
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

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, int operation_type)
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

    func(a->data, result->data, a->gpu_offset, a->shape, a->strides, a->ndims, a->total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Square root operation failed: %s", cudaGetErrorString(status));
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

    size_t new_size = 1;
    for (int i = 0; i < new_ndims; i++)
    {
        if (new_shape[i] <= 0)
        {
            return NULL;
        }
        new_size *= new_shape[i];
    }

    if (original_size != new_size)
    {
        php_error_docref(NULL, E_WARNING, "Reshape requires that the number of elements remains the same. Original: %zu, New: %zu.", original_size, new_size);
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
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    tensor_t *reshaped = cuda_tensor_create_view(
        original,
        new_shape,
        new_strides,
        new_ndims,
        original->gpu_offset,
        original->total_size
    );

    return reshaped;
}

tensor_t *cuda_tensor_transpose(tensor_t *tensor)
{
    if (!cuda_initialized() || tensor == NULL)
    {
        return NULL;
    }

    if (tensor->ndims <= 1)
    {
        return cuda_tensor_copy(tensor);
    }

    int new_shape[MAX_DIMS];
    size_t new_strides[MAX_DIMS];
    int ndims = tensor->ndims;

    for (int i = 0; i < ndims; i++)
    {
        int original_idx = ndims - 1 - i;
        new_shape[i] = tensor->shape[original_idx];
        new_strides[i] = tensor->strides[original_idx];
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

tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b)
{
    php_error_docref(NULL, E_ERROR, "Matmul not implemented yet.");
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