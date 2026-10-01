#include "tensor_transfer.h"
#include "contiguous_array_ce.h"

static void build_php_array(zval *result, const void *data, int dim, const tensor_t *tensor, size_t offset)
{
    array_init(result);
    int size = tensor->shape[dim];
    size_t stride = tensor->strides[dim];

    for (int i = 0; i < size; i++)
    {
        size_t child_offset = offset + i * stride;

        if (dim == tensor->ndims - 1)
        {
            zval val;
            switch (tensor->dtype)
            {
            case DTYPE_FLOAT32:
                ZVAL_DOUBLE(&val, (double)((const float *)data)[child_offset]);
                break;
            case DTYPE_FLOAT64:
                ZVAL_DOUBLE(&val, ((const double *)data)[child_offset]);
                break;
            case DTYPE_INT8:
                ZVAL_LONG(&val, (zend_long)((const int8_t *)data)[child_offset]);
                break;
            case DTYPE_INT16:
                ZVAL_LONG(&val, (zend_long)((const int16_t *)data)[child_offset]);
                break;
            case DTYPE_INT32:
                ZVAL_LONG(&val, (zend_long)((const int32_t *)data)[child_offset]);
                break;
            case DTYPE_INT64:
                ZVAL_LONG(&val, (zend_long)((const int64_t *)data)[child_offset]);
                break;
            case DTYPE_UINT8:
                ZVAL_LONG(&val, (zend_long)((const uint8_t *)data)[child_offset]);
                break;
            case DTYPE_UINT16:
                ZVAL_LONG(&val, (zend_long)((const uint16_t *)data)[child_offset]);
                break;
            case DTYPE_UINT32:
                ZVAL_LONG(&val, (zend_long)((const uint32_t *)data)[child_offset]);
                break;
            case DTYPE_UINT64:
                ZVAL_LONG(&val, (zend_long)((const uint64_t *)data)[child_offset]);
                break;
            case DTYPE_BOOL:
                ZVAL_BOOL(&val, ((const bool *)data)[child_offset]);
                break;
            default:
                ZVAL_NULL(&val);
                break;
            }
            zend_hash_index_update(Z_ARRVAL_P(result), i, &val);
        }
        else
        {
            zval sub;
            build_php_array(&sub, data, dim + 1, tensor, child_offset);
            zend_hash_index_update(Z_ARRVAL_P(result), i, &sub);
        }
    }
}

void tensor_to_php_array(zval *result, const tensor_t *tensor)
{
    const tensor_t *base = tensor->is_view ? tensor->base_tensor : tensor;
    void *host_data = emalloc(base->total_size * tensor->element_size);

    cudaError_t status = cudaMemcpy(
        host_data,
        base->data,
        base->total_size * tensor->element_size,
        cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        efree(host_data);
        zend_throw_error(NULL, "GPU Copy Failed: %s", cudaGetErrorString(status));
        return;
    }

    size_t offset_elements = tensor->offset / tensor->element_size;
    build_php_array(result, host_data, 0, tensor, offset_elements);
    efree(host_data);
}

tensor_t *tensor_copy_to_host(const tensor_t *tensor)
{
    tensor_t *host_tensor = ecalloc(1, sizeof(tensor_t));
    host_tensor->dtype = tensor->dtype;
    host_tensor->ndims = tensor->ndims;
    host_tensor->element_size = dtype_to_size(tensor->dtype);

    if (tensor->ndims > 0)
    {
        host_tensor->shape = emalloc(sizeof(int) * tensor->ndims);
        memcpy(host_tensor->shape, tensor->shape, sizeof(int) * tensor->ndims);

        host_tensor->strides = emalloc(sizeof(size_t) * tensor->ndims);
        memcpy(host_tensor->strides, tensor->strides, sizeof(size_t) * tensor->ndims);
    }

    host_tensor->total_size = 1;
    for (int i = 0; i < tensor->ndims; i++)
    {
        host_tensor->total_size *= tensor->shape[i];
    }

    host_tensor->allocated_size = host_tensor->total_size * host_tensor->element_size;
    host_tensor->data = allocate_for_dtype(host_tensor->dtype, host_tensor->total_size);
    if (!host_tensor->data)
    {
        if (host_tensor->shape)
            efree(host_tensor->shape);
        if (host_tensor->strides)
            efree(host_tensor->strides);
        efree(host_tensor);
        zend_throw_error(NULL, "Failed to allocate host memory");
        return NULL;
    }

    cudaError_t status = cudaMemcpy(host_tensor->data, tensor->data,
                                   host_tensor->allocated_size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        efree(host_tensor->data);
        if (host_tensor->shape)
            efree(host_tensor->shape);
        if (host_tensor->strides)
            efree(host_tensor->strides);
        efree(host_tensor);
        zend_throw_error(NULL, "CUDA error copying data to host: %s", cudaGetErrorString(status));
        return NULL;
    }

    host_tensor->is_on_gpu = 0;
    host_tensor->ref_count = 1;
    return host_tensor;
}