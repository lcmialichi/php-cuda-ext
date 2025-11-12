#include "cuda_array_wrapper.h"
#include <stdlib.h>
#include "helpers.c"
#include <string.h>
#include <cstdio>
#include "php.h"

static cudnnHandle_t cudnn_handle = NULL;
static cublasHandle_t cublas_handle = NULL;
static int cuda_initialized = 0;

int cuda_wrapper_init()
{
    if (cuda_initialized)
        return 1;

    cudaError_t cuda_status = cudaSuccess;
    cudnnStatus_t cudnn_status = CUDNN_STATUS_SUCCESS;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;

    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess)
    {
        return 0;
    }

    cudnn_status = cudnnCreate(&cudnn_handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS)
    {
        return 0;
    }

    cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        cudnnDestroy(cudnn_handle);
        return 0;
    }

    cuda_initialized = 1;
    return 1;
}

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, int operation_type)
{
    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
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

    tensor_t *result_tensor = perform_broadcast_operation(a, b, operation_type);
    if (!result_tensor)
    {
        char *shape_a = tensor_shape_as_string(a);
        char *shape_b = tensor_shape_as_string(b);
        zend_throw_error(NULL, "Broadcast failed: shapes %s and %s are incompatible",
                         shape_a,
                         shape_b);
        return NULL;
    }

    cudaError_t status = cudaDeviceSynchronize();
    return (status == cudaSuccess) ? result_tensor : NULL;
}

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

    tensor_t *result = cuda_tensor_create_empty(result_shape, result_dims);
    if (!result)
        return NULL;

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

tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims)
{
    size_t total_size = 1;
    for (int i = 0; i < ndims; i++)
    {
        total_size *= shape[i];
    }

    float *host_data = (float *)emalloc(total_size * sizeof(float));
    for (size_t i = 0; i < total_size; i++)
    {
        host_data[i] = value;
    }

    tensor_t *tensor = cuda_tensor_create(shape, ndims, host_data);
    efree(host_data);

    return tensor;
}

tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, float value)
{
    size_t total_size = 1;
    for (int i = 0; i < ndims; i++)
    {
        total_size *= shape[i];
    }

    float *data = (float *)emalloc(total_size * sizeof(float));
    for (size_t i = 0; i < total_size; i++)
    {
        data[i] = value;
    }

    tensor_t *tensor = cuda_tensor_create(shape, ndims, data);
    efree(data);

    return tensor;
}

tensor_t *cuda_tensor_create(const int shape[], int ndims, const float data[])
{
    if (!cuda_initialized && !cuda_wrapper_init())
    {
        printf("not initialized");
        return NULL;
    }

    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
        return NULL;

    tensor->ndims = ndims;
    tensor->shape = (int *)emalloc(ndims * sizeof(int));
    memcpy(tensor->shape, shape, ndims * sizeof(int));

    tensor->total_size = 1;
    tensor->ref_count = 1;

    for (int i = 0; i < ndims; i++)
    {
        tensor->total_size *= shape[i];
    }

    cudaMalloc(&tensor->data, tensor->total_size * sizeof(float));
    if (data)
    {
        cudaMemcpy(tensor->data, data, tensor->total_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    cudnnCreateTensorDescriptor(&tensor->desc);

    if (ndims <= 4)
    {
        int dims[4] = {1, 1, 1, 1};
        int strides[4] = {1, 1, 1, 1};

        for (int i = 0; i < ndims && i < 4; i++)
        {
            dims[i] = shape[i];
        }

        strides[3] = 1;
        if (ndims >= 4)
            strides[2] = dims[3];
        if (ndims >= 3)
            strides[1] = dims[2] * strides[2];
        if (ndims >= 2)
            strides[0] = dims[1] * strides[1];

        cudnnSetTensorNdDescriptor(tensor->desc, CUDNN_DATA_FLOAT, 4, dims, strides);
    }
    else
    {
        int *strides = (int *)emalloc(ndims * sizeof(int));

        strides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        cudnnSetTensorNdDescriptor(tensor->desc, CUDNN_DATA_FLOAT, ndims, shape, strides);
        efree(strides);
    }

    return tensor;
}

tensor_t *cuda_tensor_multiply(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_MUL);
}

tensor_t *cuda_tensor_subtract(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_SUB);
}

tensor_t *cuda_tensor_divide(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_DIV);
}

tensor_t *cuda_tensor_add(tensor_t *a, tensor_t *b)
{
   return cuda_tensor_op(a, b, OP_ADD);
}

tensor_t *cuda_tensor_transpose(tensor_t *tensor)
{
    if (!cuda_initialized || tensor == NULL)
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
    if (!cuda_initialized)
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

tensor_t *cuda_tensor_power(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_POW);
}

tensor_t *cuda_tensor_add_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_add_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar addition failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_subtract_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_subtract_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar subtraction failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_multiply_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_multiply_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar multiplication failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_divide_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    if (scalar == 0.0f)
    {
        php_error_docref(NULL, E_WARNING, "Division by zero");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_divide_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar division failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_power_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_power_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar power operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_sqrt(tensor_t *tensor)
{
    if (!cuda_initialized || tensor == NULL)
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
    if (!cuda_initialized || tensor == NULL)
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
    if (!cuda_initialized || tensor == NULL)
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
    if (!cuda_initialized || tensor == NULL)
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
    if (!cuda_initialized || tensor == NULL)
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

tensor_t *cuda_tensor_greater(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_GT);
}

tensor_t *cuda_tensor_less(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_LE);
}

tensor_t *cuda_tensor_equal(tensor_t *a, tensor_t *b)
{
     return cuda_tensor_op(a, b, OP_EQ);
}

tensor_t *cuda_tensor_not_equal(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_NE);
}

tensor_t *cuda_tensor_greater_equal(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_GE);
}

tensor_t *cuda_tensor_less_equal(tensor_t *a, tensor_t *b)
{
    return cuda_tensor_op(a, b, OP_LE);
}

tensor_t *cuda_tensor_greater_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_greater_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar greater than operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_less_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_less_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar less than operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_equal_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_equal_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar equal operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_not_equal_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_not_equal_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar not equal operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_greater_equal_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_greater_equal_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar greater equal operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_less_equal_scalar(tensor_t *a, float scalar)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(a->shape, a->ndims);
    if (!result)
    {
        php_error_docref(NULL, E_WARNING, "Failed to create result tensor");
        return NULL;
    }

    size_t total_size = cuda_tensor_size(a);
    launch_scalar_less_equal_kernel(a->data, scalar, result->data, total_size);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        php_error_docref(NULL, E_WARNING, "Scalar less equal operation failed: %s", cudaGetErrorString(status));
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

void cuda_tensor_destroy(tensor_t *tensor)
{
    if (!tensor)
        return;

    if (tensor->data)
        cudaFree(tensor->data);
    if (tensor->shape)
        efree(tensor->shape);
    if (tensor->desc)
        cudnnDestroyTensorDescriptor(tensor->desc);

    efree(tensor);
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

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create(shape, ndims, NULL);
}
