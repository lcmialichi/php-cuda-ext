#include "tensor_fabric.h"

static void flatten_php_array(zval *data, float *flat_array, int *index);
static void extract_shape_from_array(zval *data, int *shape, int *ndims);
static size_t calculate_total_size(zval *data);
static void flatten_php_array_to_buffer(zval *data, float *buffer, int *index);
static cudaError_t cuda_flatten_php_array_to_gpu(zval *data, float *gpu_data, int *index, size_t total_size);

tensor_t *create_tensor_from_php_array(zval *data)
{
    int shape[10] = {0};
    int ndims = 0;

    extract_shape_from_array(data, shape, &ndims);

    if (ndims == 0)
    {
        zend_throw_error(NULL, "Invalid array: cannot determine dimensions");
        return NULL;
    }

    tensor_t *tensor = cuda_tensor_create_empty(shape, ndims);
    if (!tensor)
    {
        zend_throw_error(NULL, "Failed to create empty tensor");
        return NULL;
    }

    size_t total_size = calculate_total_size(data);
    int index = 0;

    cudaError_t cuda_status = cuda_flatten_php_array_to_gpu(
        data,
        tensor->data,
        &index,
        total_size);

    if (cuda_status != cudaSuccess)
    {
        cuda_tensor_destroy(tensor);
        zend_throw_error(NULL, "Failed to copy data to GPU: %s", cudaGetErrorString(cuda_status));
        return NULL;
    }

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

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create(shape, ndims, NULL);
}

tensor_t *cuda_tensor_create(const int shape[], int ndims, const float data[])
{
    if (!tensor_init())
        return NULL;

    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
        return NULL;

    tensor->ndims = ndims;
    tensor->shape = (int *)emalloc(ndims * sizeof(int));
    memcpy(tensor->shape, shape, ndims * sizeof(int));

    tensor->strides = (size_t *)emalloc(ndims * sizeof(size_t));

    size_t stride = 1;
    for (int i = ndims - 1; i >= 0; i--)
    {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    tensor->total_size = stride;
    tensor->is_view = 0;
    tensor->gpu_offset = 0;
    tensor->slices = NULL;
    tensor->num_slices = 0;
    tensor->ref_count = 1;

    cudaMalloc((void **)&tensor->data, tensor->total_size * sizeof(float));

    if (data)
    {
        cudaMemcpy(tensor->data, data,
                   tensor->total_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    cudnnCreateTensorDescriptor(&tensor->desc);

    if (ndims <= 4)
    {
        int dims[4] = {1, 1, 1, 1};
        int strides_cudnn[4] = {1, 1, 1, 1};

        for (int i = 0; i < ndims; i++)
            dims[i] = shape[i];

        strides_cudnn[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            strides_cudnn[i] = strides_cudnn[i + 1] * dims[i + 1];

        cudnnSetTensorNdDescriptor(tensor->desc, CUDNN_DATA_FLOAT, 4, dims, strides_cudnn);
    }
    else
    {
        int *strides_cudnn = (int *)emalloc(ndims * sizeof(int));

        strides_cudnn[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; i--)
            strides_cudnn[i] = strides_cudnn[i + 1] * shape[i + 1];

        cudnnSetTensorNdDescriptor(tensor->desc, CUDNN_DATA_FLOAT, ndims, shape, strides_cudnn);
        efree(strides_cudnn);
    }

    return tensor;
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

tensor_t *resolve_result_tensor(tensor_t *t)
{
    if (t->is_view)
    {
        t->ref_count++;
        t->base_tensor->ref_count++;
        return t;
    }

    return cuda_tensor_create_empty(t->shape, t->ndims);
}

static cudaError_t cuda_flatten_php_array_to_gpu(zval *data, float *gpu_data, int *index, size_t total_size)
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

static void flatten_php_array(zval *data, float *flat_array, int *index)
{
    if (Z_TYPE_P(data) != IS_ARRAY)
    {
        if (Z_TYPE_P(data) == IS_LONG)
        {
            flat_array[(*index)++] = (float)Z_LVAL_P(data);
        }
        else if (Z_TYPE_P(data) == IS_DOUBLE)
        {
            flat_array[(*index)++] = (float)Z_DVAL_P(data);
        }
        else if (Z_TYPE_P(data) == IS_TRUE)
        {
            flat_array[(*index)++] = 1.0f;
        }
        else if (Z_TYPE_P(data) == IS_FALSE)
        {
            flat_array[(*index)++] = 0.0f;
        }
        return;
    }

    HashTable *ht = Z_ARRVAL_P(data);
    zval *current;
    ZEND_HASH_FOREACH_VAL(ht, current)
    {
        flatten_php_array(current, flat_array, index);
    }
    ZEND_HASH_FOREACH_END();
}

static void extract_shape_from_array(zval *data, int *shape, int *ndims)
{
    *ndims = 0;

    void extract_shape_recursive(zval * arr, int current_depth)
    {
        if (Z_TYPE_P(arr) != IS_ARRAY)
            return;
        if (current_depth >= 10)
            return;

        HashTable *arr_ht = Z_ARRVAL_P(arr);
        int count = zend_array_count(arr_ht);

        if (count == 0)
            return;

        shape[current_depth] = count;
        if (current_depth >= *ndims)
        {
            *ndims = current_depth + 1;
        }

        if (count > 0)
        {
            zval *first = zend_hash_index_find(arr_ht, 0);
            if (first != NULL)
            {
                extract_shape_recursive(first, current_depth + 1);
            }
        }
    }

    extract_shape_recursive(data, 0);
}

static size_t calculate_total_size(zval *data)
{
    if (Z_TYPE_P(data) != IS_ARRAY)
    {
        return 1;
    }

    size_t total = 1;
    HashTable *ht = Z_ARRVAL_P(data);
    zval *first = zend_hash_index_find(ht, 0);

    if (first != NULL)
    {
        total = zend_array_count(ht) * calculate_total_size(first);
    }

    return total;
}