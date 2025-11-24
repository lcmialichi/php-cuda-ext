#include "tensor_fabric.h"
#include "cuda_kernels.h"
#include "memory_pool.h"

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

int cuda_tensor_get_scalar_value(tensor_t *scalar_tensor, float *result_val)
{
    if (scalar_tensor->ndims != 0)
    {
        return FAILURE;
    }

    void *gpu_source_ptr;

    if (scalar_tensor->is_view)
    {
        if (!scalar_tensor->base_tensor)
        {
            zend_error(E_WARNING, "Scalar view has no base tensor.");
            return FAILURE;
        }
        gpu_source_ptr = (char *)scalar_tensor->base_tensor->data + scalar_tensor->gpu_offset;
    }
    else
    {
        gpu_source_ptr = scalar_tensor->data;
    }

    cudaError_t status = cudaMemcpy(
        result_val,
        gpu_source_ptr,
        sizeof(float),
        cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        zend_error(E_WARNING, "Failed to copy scalar data from GPU: %s", cudaGetErrorString(status));
        return FAILURE;
    }

    return SUCCESS;
}

tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, float value)
{
    tensor_t *tensor = cuda_tensor_create_empty(shape, ndims);
    if (!tensor)
    {
        return NULL;
    }

    launch_fill_kernel(tensor->data, value, tensor->total_size);
    return tensor;
}

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create_float(shape, ndims, NULL);
}

tensor_t *cuda_tensor_create(const int shape[], int ndims, const void *data, int dtype)
{
    if (!tensor_init())
        return NULL;

    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
        return NULL;

    size_t element_size;
    if (dtype == DTYPE_FLOAT)
    {
        element_size = sizeof(float);
    }
    else if (dtype == DTYPE_INT)
    {
        element_size = sizeof(int);
    }
    else
    {
        efree(tensor);
        zend_throw_error(NULL, "Unsupported data type for tensor creation: %d", dtype);
        return NULL;
    }

    tensor->dtype = dtype;

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

    int *d_shape;
    size_t *d_strides;

    cudaMalloc((void **)&d_shape, ndims * sizeof(int));
    cudaMalloc((void **)&d_strides, ndims * sizeof(size_t));
    cudaMemcpy(d_shape, tensor->shape, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, tensor->strides, ndims * sizeof(size_t), cudaMemcpyHostToDevice);

    tensor->total_size = stride;
    tensor->is_view = 0;
    tensor->gpu_offset = 0;
    tensor->slices = NULL;
    tensor->num_slices = 0;
    tensor->ref_count = 1;
    tensor->d_shape = d_shape;
    tensor->d_strides = d_strides;

    size_t required_bytes = tensor->total_size * element_size;
    tensor->allocated_size = required_bytes;

    tensor->data = tensor_mem_alloc(required_bytes);

    if (!tensor->data)
    {
        if (tensor->strides)
            efree(tensor->strides);
        if (tensor->shape)
            efree(tensor->shape);
        cudaFree(d_shape);
        cudaFree(d_strides);
        efree(tensor);
        zend_throw_error(NULL, "Failed to allocate GPU memory for tensor.");
        return NULL;
    }

    if (data)
    {
        cudaError_t err = cudaMemcpy(tensor->data, data,
                                     required_bytes,
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            tensor_mem_free(tensor->data);
            cudaFree(d_shape);
            cudaFree(d_strides);
            efree(tensor->strides);
            efree(tensor->shape);
            efree(tensor);
            zend_throw_error(NULL, "Failed to copy data to GPU: %s", cudaGetErrorString(err));
            return NULL;
        }
    }

    return tensor;
}

tensor_t *cuda_tensor_create_float(const int shape[], int ndims, const float data[])
{
    return cuda_tensor_create(shape, ndims, data, DTYPE_FLOAT);
}

tensor_t *cuda_tensor_create_int(const int shape[], int ndims, const int data[])
{
    return cuda_tensor_create(shape, ndims, data, DTYPE_INT);
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

    tensor_t *tensor = cuda_tensor_create_float(shape, ndims, host_data);
    efree(host_data);

    return tensor;
}

tensor_t *resolve_result_tensor(tensor_t *t)
{
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

tensor_t *cuda_tensor_clone(tensor_t *base_tensor)
{
    if (base_tensor == NULL)
    {
        return NULL;
    }

    size_t total_elements = base_tensor->total_size;
    size_t total_bytes = total_elements * sizeof(float);

    tensor_t *new_tensor = cuda_tensor_create_float(base_tensor->shape, base_tensor->ndims, base_tensor->data);

    if (new_tensor == NULL || new_tensor->data == NULL)
    {
        if (new_tensor)
            cuda_tensor_destroy(new_tensor);
        return NULL;
    }
    return new_tensor;
}