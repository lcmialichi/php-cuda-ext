#include "tensor_fabric.h"
#include "cuda_kernels.h"
#include "memory_pool.h"
#include <time.h>
#include <curand.h>
#include "data_types.h"
#include "factory_kernels.h"
#include <stdbool.h>

static void flatten_php_array(zval *data, float *flat_array, int *index);
static void extract_shape_from_array(zval *data, int *shape, int *ndims);
static size_t calculate_total_size(zval *data);
static cudaError_t cuda_flatten_php_array_to_gpu(zval *data, void *gpu_data, int *index, size_t total_size, dtype_t dtype);

tensor_t *tensor_cast_string(tensor_t *tensor, const char *new_dtype_str)
{
    if (!tensor || !new_dtype_str)
    {
        return NULL;
    }

    dtype_t new_dtype = dtype_from_string(new_dtype_str);
    if (new_dtype >= DTYPE_COUNT || new_dtype == DTYPE_UNKNOWN)
    {
        php_error_docref(NULL, E_WARNING, "Invalid dtype string: %s", new_dtype_str);
        return NULL;
    }

    return tensor_cast(tensor, new_dtype);
}

tensor_t *create_tensor_from_php_array(zval *data, dtype_t dtype)
{
    int shape[10] = {0};
    int ndims = 0;

    extract_shape_from_array(data, shape, &ndims);

    if (ndims == 0)
    {
        zend_throw_error(NULL, "Invalid array: cannot determine dimensions");
        return NULL;
    }

    tensor_t *tensor = cuda_tensor_create_empty_with_dtype(shape, ndims, dtype);
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
        total_size,
        dtype);

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

tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, scalar_value_t value, dtype_t dtype)
{
    scalar_value_t scalar_value = cast_single_value(value, dtype);
    if (scalar_value.dtype == DTYPE_UNKNOWN)
    {
        zend_throw_error(NULL, "Failed create CudaArray object with dtype: %s, received %s as value.",
                         dtype_to_string(dtype),
                         dtype_to_string(value.dtype));

        return NULL;
    }

    tensor_t *tensor = cuda_tensor_create_empty_with_dtype(shape, ndims, dtype);
    if (!tensor)
    {
        return NULL;
    }

    launch_assign_scalar_val_kernel(tensor->data, tensor->dtype, scalar_value, tensor->total_size);
    return tensor;
}

void destroy_curand_generator(curandGenerator_t generator)
{
    if (generator)
    {
        curandDestroyGenerator(generator);
    }
}

int set_rand_tensor_data(void *data,
                         size_t size,
                         unsigned long long seed,
                         scalar_value_t min_value,
                         scalar_value_t max_value,
                         dtype_t dtype)
{
    curandGenerator_t generator = NULL;
    curandStatus_t status;

    float *temp = cuda_mem_alloc(sizeof(float) * size);
    if (temp == NULL)
    {
        zend_throw_error(NULL, "CUDA Out of Memory: Failed to allocate temporary buffer for Random.");
        return FAILURE;
    }

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

    status = curandGenerateUniform(generator, temp, size);
    if (status != CURAND_STATUS_SUCCESS)
    {
        destroy_curand_generator(generator);
        return FAILURE;
    }

    cudaDeviceSynchronize();

    if (dtype == DTYPE_BOOL)
    {
        launch_bernoulli_kernel(temp, data, size, 0.5f);
    }
    else
    {
        launch_scale_range_kernel(temp, data, dtype, min_value, max_value, size);
    }

    cudaError_t err = cudaDeviceSynchronize();

    destroy_curand_generator(generator);
    cuda_mem_free(temp);

    if (err != cudaSuccess)
    {
        zend_throw_error(NULL, "Kernel failed: %s", cudaGetErrorString(err));
        return FAILURE;
    }

    return SUCCESS;
}

tensor_t *cuda_tensor_create_rand(
    int *shape,
    int ndims,
    scalar_value_t min_value,
    scalar_value_t max_value,
    dtype_t dtype,
    unsigned long long seed)
{
    tensor_t *tensor = cuda_tensor_create_empty_with_dtype(shape, ndims, dtype);
    if (!tensor)
    {
        zend_throw_error(NULL, "Unable to create random tensor.");
        return NULL;
    }

    if (can_cast_unsafe(min_value.dtype, dtype) != 1 || can_cast_unsafe(max_value.dtype, dtype) != 1)
    {
        zend_throw_error(NULL, "Failed to cast min and max value to dtype: %s.", dtype_to_string(dtype));
        return NULL;
    }

    scalar_value_t casted_min = cast_single_value(min_value, dtype);
    scalar_value_t casted_max = cast_single_value(max_value, dtype);

    if (set_rand_tensor_data(
            tensor->data,
            tensor->total_size,
            seed,
            casted_min,
            casted_max,
            dtype) != SUCCESS)
    {

        zend_throw_error(NULL, "Kernel failed.");
        cuda_tensor_destroy(tensor);
        return NULL;
    }

    return tensor;
}

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create_float(shape, ndims, NULL);
}

tensor_t *cuda_tensor_create_empty_dtype(const int shape[], int ndims, dtype_t dtype)
{
    return cuda_tensor_create(shape, ndims, NULL, dtype);
}

tensor_t *cuda_tensor_create(const int shape[], int ndims, const void *data, dtype_t dtype)
{
    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
        return NULL;

    size_t element_size = dtype_size(dtype);
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

    int *d_shape = cuda_mem_alloc(ndims * sizeof(int));
    size_t *d_strides = cuda_mem_alloc(ndims * sizeof(size_t));
    cudaMemcpy(d_shape, tensor->shape, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, tensor->strides, ndims * sizeof(size_t), cudaMemcpyHostToDevice);

    tensor->total_size = stride;
    tensor->is_view = 0;
    tensor->offset = 0;
    tensor->slices = NULL;
    tensor->num_slices = 0;
    tensor->ref_count = 1;
    tensor->d_shape = d_shape;
    tensor->d_strides = d_strides;
    tensor->element_size = element_size;
    tensor->is_on_gpu = 1;
    tensor->is_contiguous_cached = -1;

    size_t required_bytes = tensor->total_size * element_size;
    tensor->allocated_size = required_bytes;

    tensor->data = cuda_mem_alloc(required_bytes);

    if (!tensor->data)
    {
        if (tensor->strides)
            efree(tensor->strides);
        if (tensor->shape)
            efree(tensor->shape);
        cuda_mem_free(d_shape);
        cuda_mem_free(d_strides);
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
            cuda_mem_free(tensor->data);
            cuda_mem_free(d_shape);
            cuda_mem_free(d_strides);
            efree(tensor->strides);
            efree(tensor->shape);
            efree(tensor);
            zend_throw_error(NULL, "Failed to copy data to GPU: %s", cudaGetErrorString(err));
            return NULL;
        }
    }

    return tensor;
}

tensor_t *cuda_tensor_create_on_host(const int shape[], int ndims, void *data, dtype_t dtype)
{
    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
        return NULL;

    size_t element_size = dtype_size(dtype);

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

    tensor->total_size = stride;
    tensor->is_view = 0;
    tensor->offset = 0;
    tensor->slices = NULL;
    tensor->num_slices = 0;
    tensor->ref_count = 1;
    tensor->d_shape = NULL;
    tensor->d_strides = NULL;
    tensor->element_size = element_size;
    tensor->is_on_gpu = 0;
    tensor->is_contiguous_cached = -1;

    size_t required_bytes = tensor->total_size * element_size;
    tensor->allocated_size = required_bytes;

    tensor->data = emalloc(required_bytes);
    if (!tensor->data)
    {
        efree(tensor->strides);
        efree(tensor->shape);
        efree(tensor);
        zend_throw_error(NULL, "Failed to allocate Host memory for tensor.");
        return NULL;
    }

    if (data)
    {
        memcpy(tensor->data, data, required_bytes);
    }
    else
    {
        memset(tensor->data, 0, required_bytes);
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
    return cuda_tensor_create_empty_dtype(t->shape, t->ndims, t->dtype);
}

static cudaError_t cuda_flatten_php_array_to_gpu(zval *data, void *gpu_data, int *index, size_t total_size, dtype_t dtype)
{
    void *pinned_host_data;
    size_t el_size = dtype_size(dtype);

    cudaError_t status = cudaMallocHost(&pinned_host_data, total_size * el_size);
    if (status != cudaSuccess)
        return status;

    int host_index = 0;

    switch (dtype)
    {
    case DTYPE_FLOAT32:
        flatten_php_array_to_float32(data, (float *)pinned_host_data, &host_index);
        break;
    case DTYPE_FLOAT64:
        flatten_php_array_to_float64(data, (double *)pinned_host_data, &host_index);
        break;
    case DTYPE_INT32:
        flatten_php_array_to_int32(data, (int32_t *)pinned_host_data, &host_index);
        break;
    case DTYPE_INT8:
        flatten_php_array_to_int8(data, (int8_t *)pinned_host_data, &host_index);
        break;
    case DTYPE_INT64:
        flatten_php_array_to_int64(data, (int64_t *)pinned_host_data, &host_index);
        break;
    case DTYPE_BOOL:
        flatten_php_array_to__bool(data, (bool *)pinned_host_data, &host_index);
        break;
    default:
        cudaFreeHost(pinned_host_data);
        return cudaErrorInvalidValue;
    }

    status = cudaMemcpy(gpu_data, pinned_host_data, total_size * el_size, cudaMemcpyHostToDevice);

    cudaFreeHost(pinned_host_data);
    *index = host_index;
    return status;
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