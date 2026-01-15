#include "tensor_fabric.h"
#include "cuda_kernels.h"
#include "memory_pool.h"
#include <time.h>
#include <curand.h>
#include "data_types.h"

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

int cuda_tensor_get_scalar_value(tensor_t *t, float *result_val, int index)
{
    size_t byte_offset = (size_t)index * t->element_size;
    void *gpu_source_ptr = (void *)((char *)t->data + byte_offset);
    if (byte_offset >= (size_t)t->total_size * t->element_size)
    {
        zend_error(E_WARNING, "Index out of bounds.");
        return FAILURE;
    }

    cudaError_t status = cudaMemcpy(
        result_val,
        gpu_source_ptr,
        t->element_size,
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

void destroy_curand_generator(curandGenerator_t generator)
{
    if (generator)
    {
        curandDestroyGenerator(generator);
    }
}

int set_rand_tensor_data(float *data, size_t size, unsigned long long seed, float min_value, float max_value)
{
    curandGenerator_t generator = NULL;
    curandStatus_t status;

    status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS)
        return FAILURE;

    if (seed == 0)
    {
        seed = (unsigned long long)time(NULL);
    }
    status = curandSetPseudoRandomGeneratorSeed(generator, seed);
    if (status != CURAND_STATUS_SUCCESS)
    {
        destroy_curand_generator(generator);
        return FAILURE;
    }

    status = curandGenerateUniform(generator, data, size);
    if (status != CURAND_STATUS_SUCCESS)
    {
        destroy_curand_generator(generator);
        return FAILURE;
    }

    if (min_value != 0.0f || max_value != 1.0f)
    {
        if (launch_scale_kernel_host(data, size, min_value, max_value) != SUCCESS)
        {
            destroy_curand_generator(generator);
            return FAILURE;
        }
    }
    return SUCCESS;
}

tensor_t *cuda_tensor_create_rand(
    int *shape,
    int ndims,
    float min_value,
    float max_value,
    unsigned long long seed)
{
    tensor_t *tensor = cuda_tensor_create_empty(shape, ndims);
    if (!tensor)
    {
        return NULL;
    }

    if (set_rand_tensor_data(
            tensor->data,
            tensor->total_size,
            seed,
            min_value,
            max_value) != SUCCESS)
    {
        cuda_tensor_destroy(tensor);
        return NULL;
    }

    return tensor;
}

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create_float(shape, ndims, NULL);
}

tensor_t *cuda_tensor_create(const int shape[], int ndims, const void *data, dtype_t dtype)
{
    if (!tensor_init())
        return NULL;

    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
        return NULL;

    size_t element_size;
    if (dtype == DTYPE_FLOAT32)
    {
        element_size = sizeof(float);
    }
    else if (dtype == DTYPE_INT32)
    {
        element_size = sizeof(int);
    }
    else
    {
        efree(tensor);
        zend_throw_error(NULL, "Unsupported data type for tensor creation");
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
    tensor->element_size = element_size;
    tensor->is_on_gpu = 1;

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
    return cuda_tensor_create(shape, ndims, data, DTYPE_FLOAT32);
}

tensor_t *cuda_tensor_create_int(const int shape[], int ndims, const int data[])
{
    return cuda_tensor_create(shape, ndims, data, DTYPE_INT32);
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

    tensor_t *new_tensor = cuda_tensor_create(base_tensor->shape, base_tensor->ndims, base_tensor->data, base_tensor->dtype);

    if (new_tensor == NULL || new_tensor->data == NULL)
    {
        if (new_tensor)
            cuda_tensor_destroy(new_tensor);
        return NULL;
    }
    return new_tensor;
}