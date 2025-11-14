#include "cuda_array_wrapper.h"
#include "broadcast_ops.h"
#include "scalar_ops.h"
#include "operations.h"
#include <stdlib.h>
#include "helpers.c"
#include <string.h>
#include <cstdio>
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

int calculate_broadcast_shape(int *a_shape, int a_dims,
                              int *b_shape, int b_dims,
                              int *result_shape, int *result_dims)
{
    *result_dims = (a_dims > b_dims) ? a_dims : b_dims;

    for (int i = 0; i < *result_dims; i++)
    {
        int a_dim = (i < *result_dims - a_dims) ? 1 : a_shape[i - (*result_dims - a_dims)];
        int b_dim = (i < *result_dims - b_dims) ? 1 : b_shape[i - (*result_dims - b_dims)];

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

    for (int i = 0; i < a->ndims; i++)
    {
        int result_dim_idx = *result_dims - a->ndims + i;
        if (result_dim_idx >= 0 && a->shape[i] != 1)
        {
            a_strides[i] = calculate_broadcast_stride(result_shape, *result_dims, result_dim_idx);
        }
        else
        {
            a_strides[i] = 0;
        }
    }

    for (int i = 0; i < b->ndims; i++)
    {
        int result_dim_idx = *result_dims - b->ndims + i;
        if (result_dim_idx >= 0 && b->shape[i] != 1)
        {
            b_strides[i] = calculate_broadcast_stride(result_shape, *result_dims, result_dim_idx);
        }
        else
        {
            b_strides[i] = 0;
        }
    }

    return 1;
}

tensor_t *perform_broadcast_operation(tensor_t *a, tensor_t *b, int operation_type)
{
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

    tensor_t *result;

    if (a->is_view)
    {
        result = resolve_result_tensor(a);
        if (!result)
        {
            return NULL;
        }

        efree(result->shape);
        result->shape = (int *)emalloc(result_dims * sizeof(int));
        memcpy(result->shape, result_shape, result_dims * sizeof(int));

        result->ndims = result_dims;
        result->total_size = 1;
        for (int i = 0; i < result_dims; i++)
        {
            result->total_size *= result_shape[i];
        }
    }
    else
    {
        result = cuda_tensor_create_empty(result_shape, result_dims);
        if (!result)
        {
            return NULL;
        }
    }

    if (a->data == NULL || b->data == NULL || result->data == NULL)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    launch_broadcast_kernel(
        a->data, b->data, result->data,
        a_strides, a->ndims,
        b_strides, b->ndims,
        result_shape, result_dims,
        total_elements, operation_type);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

scalar_fn get_scalar_fn(int op)
{
    for (int i = 0; i < sizeof(scalar_dispatch) / sizeof(ScalarDispatchEntry); i++)
        if (scalar_dispatch[i].op == op)
            return scalar_dispatch[i].fn;

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

    if (result->desc == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Result tensor descriptor is NULL");
        cuda_tensor_destroy(result);
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
        char *shape_a = tensor_shape_as_string(a);
        char *shape_b = tensor_shape_as_string(b);
        zend_throw_error(NULL, "Broadcast failed: shapes %s and %s are incompatible",
                         shape_a,
                         shape_b);
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
         total_elements);

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

    size_t total_size = cuda_tensor_size(a);
    scalar_fn func = get_scalar_fn(operation_type);
    if (func == NULL)
    {
        php_error_docref(NULL, E_ERROR, "Operation handler not found.");
        return NULL;
    }

    func(a->data, scalar, result->data, total_size);
    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar operation failed: %s", cudaGetErrorString(status));
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
        return NULL;
    }

    tensor_t *reshaped = (tensor_t *)emalloc(sizeof(tensor_t));
    if (reshaped == NULL)
    {
        return NULL;
    }

    reshaped->data = original->data;
    reshaped->ndims = new_ndims;
    original->ref_count++;
    reshaped->ref_count = 1;

    reshaped->shape = (int *)emalloc(new_ndims * sizeof(int));
    if (reshaped->shape == NULL)
    {
        efree(reshaped);
        return NULL;
    }

    memcpy(reshaped->shape, new_shape, new_ndims * sizeof(int));

    reshaped->total_size = 1;
    for (int i = 0; i < new_ndims; i++)
    {
        reshaped->total_size *= new_shape[i];
    }

    cudnnStatus_t status = cudnnCreateTensorDescriptor(&reshaped->desc);
    if (status != CUDNN_STATUS_SUCCESS)
    {
        efree(reshaped->shape);
        efree(reshaped);
        return NULL;
    }

    if (new_ndims <= 4)
    {
        int dims[4] = {1, 1, 1, 1};
        int strides[4] = {1, 1, 1, 1};

        for (int i = 0; i < new_ndims && i < 4; i++)
        {
            dims[i] = new_shape[i];
        }

        strides[3] = 1;
        if (new_ndims >= 4)
            strides[2] = dims[3];
        if (new_ndims >= 3)
            strides[1] = dims[2] * strides[2];
        if (new_ndims >= 2)
            strides[0] = dims[1] * strides[1];

        cudnnSetTensorNdDescriptor(reshaped->desc, CUDNN_DATA_FLOAT, 4, dims, strides);
    }
    else
    {
        int *strides = (int *)emalloc(new_ndims * sizeof(int));
        if (strides == NULL)
        {
            cudnnDestroyTensorDescriptor(reshaped->desc);
            efree(reshaped->shape);
            efree(reshaped);
            return NULL;
        }

        strides[new_ndims - 1] = 1;
        for (int i = new_ndims - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * new_shape[i + 1];
        }

        cudnnSetTensorNdDescriptor(reshaped->desc, CUDNN_DATA_FLOAT, new_ndims, new_shape, strides);
        efree(strides);
    }

    return reshaped;
}

cudaError_t cuda_flatten_php_array_to_gpu(zval *data, float *gpu_data, int *index, size_t total_size)
{
    float *pinned_host_data;
    cudaError_t status = cudaMallocHost((void **)&pinned_host_data, total_size * sizeof(float));
    if (status != cudaSuccess)
        return status;

    int host_index = 0;
    flatten_php_array_to_buffer(data, pinned_host_data, &host_index);

    status = cudaMemcpyAsync(gpu_data, pinned_host_data, total_size * sizeof(float),
                             cudaMemcpyHostToDevice, 0);

    cudaFreeHost(pinned_host_data);
    *index = host_index;
    return status;
}

static void flatten_php_array_to_buffer(zval *data, float *buffer, int *index)
{
    if (Z_TYPE_P(data) == IS_ARRAY)
    {
        HashTable *ht = Z_ARRVAL_P(data);
        zval *current;
        ZEND_HASH_FOREACH_VAL(ht, current)
        {
            flatten_php_array_to_buffer(current, buffer, index);
        }
        ZEND_HASH_FOREACH_END();
        return;
    }

    if (Z_TYPE_P(data) == IS_LONG)
    {
        buffer[(*index)++] = (float)Z_LVAL_P(data);
    }
    else if (Z_TYPE_P(data) == IS_DOUBLE)
    {
        buffer[(*index)++] = (float)Z_DVAL_P(data);
    }
    else if (Z_TYPE_P(data) == IS_TRUE)
    {
        buffer[(*index)++] = 1.0f;
    }
    else if (Z_TYPE_P(data) == IS_FALSE)
    {
        buffer[(*index)++] = 0.0f;
    }
}

tensor_t *cuda_tensor_transpose(tensor_t *tensor)
{
    if (!cuda_initialized() || tensor == NULL)
    {
        return NULL;
    }

    if (tensor->ndims == 1)
    {
        php_error_docref(NULL, E_WARNING, "Dim = 1");
        return cuda_tensor_copy(tensor);
    }

    if (tensor->ndims == 2)
    {
        int new_shape[2] = {tensor->shape[1], tensor->shape[0]};
        tensor_t *result = cuda_tensor_create_empty(new_shape, 2);
        if (!result)
            return NULL;

        float alpha = 1.0f;
        float beta = 0.0f;

        cublasStatus_t status = cublasSgeam(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            new_shape[1],
            new_shape[0],
            &alpha,
            tensor->data, tensor->shape[1],
            &beta,
            tensor->data, tensor->shape[1],
            result->data, new_shape[1]);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            cuda_tensor_destroy(result);
            return NULL;
        }

        return result;
    }

    php_error_docref(NULL, E_WARNING, "Transpose not implemented for tensors > 2D");
    return NULL;
}

tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b)
{
    if (!cuda_initialized())
        return NULL;

    int a_ndims = a->ndims;
    int b_ndims = b->ndims;

    int a_rows, a_cols, b_rows, b_cols;

    if (a_ndims == 1 && b_ndims == 1)
    {
        if (a->shape[0] != b->shape[0])
            return NULL;
        a_rows = 1;
        a_cols = a->shape[0];
        b_rows = b->shape[0];
        b_cols = 1;
    }
    else if (a_ndims == 1 && b_ndims == 2)
    {
        if (a->shape[0] != b->shape[0])
            return NULL;
        a_rows = 1;
        a_cols = a->shape[0];
        b_rows = b->shape[0];
        b_cols = b->shape[1];
    }
    else if (a_ndims == 2 && b_ndims == 1)
    {
        if (a->shape[1] != b->shape[0])
            return NULL;
        a_rows = a->shape[0];
        a_cols = a->shape[1];
        b_rows = b->shape[0];
        b_cols = 1;
    }
    else if (a_ndims == 2 && b_ndims == 2)
    {
        if (a->shape[1] != b->shape[0])
            return NULL;
        a_rows = a->shape[0];
        a_cols = a->shape[1];
        b_rows = b->shape[0];
        b_cols = b->shape[1];
    }
    else
    {
        return NULL;
    }

    int result_rows = a_rows;
    int result_cols = b_cols;
    int result_ndims = (a_ndims == 1 && b_ndims == 1) ? 1 : 2;
    int result_shape[2] = {result_rows, result_cols};

    tensor_t *result = cuda_tensor_create_empty(result_shape, result_ndims);
    if (!result)
        return NULL;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        result_cols,
        result_rows,
        a_cols,
        &alpha,
        b->data,
        result_cols,
        a->data,
        a_cols,
        &beta,
        result->data,
        result_cols);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_sqrt(tensor_t *tensor)
{
    if (!cuda_initialized() || tensor == NULL)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(tensor->shape, tensor->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(tensor);
    launch_sqrt_kernel(tensor->data, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Square root operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_exp(tensor_t *tensor)
{
    if (!cuda_initialized() || tensor == NULL)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(tensor->shape, tensor->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(tensor);
    launch_exp_kernel(tensor->data, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Exponential operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_log(tensor_t *tensor)
{
    if (!cuda_initialized() || tensor == NULL)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(tensor->shape, tensor->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(tensor);
    launch_log_kernel(tensor->data, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Logarithm operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_sin(tensor_t *tensor)
{
    if (!cuda_initialized() || tensor == NULL)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(tensor->shape, tensor->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(tensor);
    launch_sin_kernel(tensor->data, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Sine operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_cos(tensor_t *tensor)
{
    if (!cuda_initialized() || tensor == NULL)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized or tensor is NULL");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(tensor->shape, tensor->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(tensor);
    launch_cos_kernel(tensor->data, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Cosine operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
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
        cuda_tensor_size(tensor) * sizeof(float),
        cudaMemcpyDeviceToDevice);

    if (cuda_status != cudaSuccess)
    {
        cuda_tensor_destroy(copy);
        return NULL;
    }

    return copy;
}

size_t cuda_tensor_size(tensor_t *tensor)
{
    if (!tensor)
        return 0;

    size_t size = 1;
    for (int i = 0; i < tensor->ndims; i++)
    {
        size *= tensor->shape[i];
    }
    return size;
}

int *calculate_strides(int *shape, int ndims)
{
    if (ndims <= 0 || !shape)
        return NULL;

    int *strides = (int *)emalloc(ndims * sizeof(int));
    if (!strides)
        return NULL;

    strides[ndims - 1] = 1;
    for (int i = ndims - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}

int *cuda_tensor_get_shape(tensor_t *tensor)
{
    return tensor->shape;
}
