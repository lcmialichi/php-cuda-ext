#include "cuda_array_wrapper.h"
#include "cuda_kernels.h"
#include <stdlib.h>
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

extern "C" { 
    void launch_divide_kernel(float *a, float *b, float *result, int n);
}

tensor_t *cuda_tensor_divide(tensor_t *a, tensor_t *b, tensor_t *result) {
    size_t total_size = cuda_tensor_size(a);
    launch_divide_kernel(a->data, b->data, result->data, total_size);
    
    cudaError_t status = cudaDeviceSynchronize();
    return (status == cudaSuccess) ? result : NULL;
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
    if (Z_TYPE_P(data) != IS_ARRAY)
    {
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
        return;
    }

    HashTable *ht = Z_ARRVAL_P(data);
    zval *current;
    ZEND_HASH_FOREACH_VAL(ht, current)
    {
        flatten_php_array_to_buffer(current, buffer, index);
    }
    ZEND_HASH_FOREACH_END();
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
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    if (a == NULL || b == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Null tensor input");
        return NULL;
    }

    if (a->data == NULL || b->data == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor data is NULL");
        return NULL;
    }

    if (a->desc == NULL || b->desc == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor descriptor is NULL");
        return NULL;
    }

    if (a->ndims != b->ndims)
    {
        php_error_docref(NULL, E_WARNING, "Different number of dimensions: %d vs %d",
                         a->ndims, b->ndims);
        return NULL;
    }

    for (int i = 0; i < a->ndims; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            php_error_docref(NULL, E_WARNING, "Shape mismatch at dimension %d: %d vs %d",
                             i, a->shape[i], b->shape[i]);
            return NULL;
        }
    }

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

    return cuda_tensor_op(a, b, result, CUDNN_OP_MUL);
}


tensor_t *cuda_tensor_divide(tensor_t *a, tensor_t *b)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    if (a == NULL || b == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Null tensor input");
        return NULL;
    }

    if (a->data == NULL || b->data == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor data is NULL");
        return NULL;
    }

    if (a->desc == NULL || b->desc == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor descriptor is NULL");
        return NULL;
    }

    if (a->ndims != b->ndims)
    {
        php_error_docref(NULL, E_WARNING, "Different number of dimensions: %d vs %d",
                         a->ndims, b->ndims);
        return NULL;
    }

    for (int i = 0; i < a->ndims; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            php_error_docref(NULL, E_WARNING, "Shape mismatch at dimension %d: %d vs %d",
                             i, a->shape[i], b->shape[i]);
            return NULL;
        }
    }

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

    return cuda_tensor_op(a, b, result, KERNEL_DIV);
}

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, tensor_t *result, cuda_op_type_t op_type)
{
    if (!cuda_initialized || !a || !b || !result) {
        return NULL;
    }

    cudnnOpTensorDescriptor_t op_desc;
    cudnnStatus_t status = cudnnCreateOpTensorDescriptor(&op_desc);
    
    if (status != CUDNN_STATUS_SUCCESS) {
        return NULL;
    }

    float alpha1 = 1.0f;
    float alpha2 = 1.0f;
    float beta = 0.0f;

    cudnnOpTensorOp_t cudnn_op;
    switch (op_type)
    {
    case CUDNN_OP_ADD:
        cudnn_op = CUDNN_OP_TENSOR_ADD;
        break;
    case CUDNN_OP_MUL:
        cudnn_op = CUDNN_OP_TENSOR_MUL;
        break;
    case CUDNN_OP_SUB:
        cudnn_op = CUDNN_OP_TENSOR_ADD;
        alpha2 = -1.0f;
        break;
    case CUDNN_OP_MIN:
        cudnn_op = CUDNN_OP_TENSOR_MIN;
        break;
    case CUDNN_OP_MAX:
        cudnn_op = CUDNN_OP_TENSOR_MAX;
        break;
    case KERNEL_DIV:
        return cuda_tensor_divide(a, b, result);
    default:
        cudnnDestroyOpTensorDescriptor(op_desc);
        return NULL;
    }

    status = cudnnSetOpTensorDescriptor(
        op_desc,
        cudnn_op,
        CUDNN_DATA_FLOAT,
        CUDNN_PROPAGATE_NAN);

    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyOpTensorDescriptor(op_desc);
        return NULL;
    }

    status = cudnnOpTensor(
        cudnn_handle,
        op_desc,
        &alpha1,
        a->desc, a->data,
        &alpha2,
        b->desc, b->data,
        &beta,
        result->desc, result->data);

    cudnnDestroyOpTensorDescriptor(op_desc);

    return (status == CUDNN_STATUS_SUCCESS) ? result : NULL;
}

tensor_t *cuda_tensor_add(tensor_t *a, tensor_t *b)
{
    if (!cuda_initialized)
    {
        php_error_docref(NULL, E_WARNING, "CUDA not initialized");
        return NULL;
    }

    if (a == NULL || b == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Null tensor input");
        return NULL;
    }

    if (a->data == NULL || b->data == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor data is NULL");
        return NULL;
    }

    if (a->desc == NULL || b->desc == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor descriptor is NULL");
        return NULL;
    }

    if (a->ndims != b->ndims)
    {
        php_error_docref(NULL, E_WARNING, "Different number of dimensions: %d vs %d",
                         a->ndims, b->ndims);
        return NULL;
    }

    for (int i = 0; i < a->ndims; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            php_error_docref(NULL, E_WARNING, "Shape mismatch at dimension %d: %d vs %d",
                             i, a->shape[i], b->shape[i]);
            return NULL;
        }
    }

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

    return cuda_tensor_op(a, b, result, CUDNN_OP_ADD);
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
