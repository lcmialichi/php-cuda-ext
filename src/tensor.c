#include "tensor.h"
#include "php.h"
#include "Zend/zend_API.h"
#include <string.h>
#include "memory_pool.h"

static int cuda_is_initialized = 0;

int tensor_init()
{
    if (cuda_initialized())
        return 1;

    cudaError_t cuda_status = cudaSuccess;

    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess)
    {
        return 0;
    }

    cuda_set_initialized(1);
    return 1;
}

int is_contiguous(tensor_t *tensor)
{
    if (tensor == NULL)
    {
        return 0;
    }

    if (tensor->ndims <= 1)
    {
        return 1;
    }

    size_t expected_stride = 1;
    int ndims = tensor->ndims;

    for (int i = ndims - 1; i >= 0; i--)
    {
        if (tensor->strides[i] != expected_stride)
        {
            return 0;
        }

        expected_stride *= tensor->shape[i];
    }

    return 1;
}

tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, int *shape, size_t *strides, int dims, size_t offset, size_t total_size)
{
    size_t byte_offset = offset * sizeof(float);

    tensor_t *view = (tensor_t *)emalloc(sizeof(tensor_t));

    memset(view, 0, sizeof(tensor_t));

    view->is_view = 1;
    view->gpu_offset = 0;
    view->data = (float *)((char *)base_tensor->data + byte_offset);
    view->total_size = total_size;
    view->ref_count = 1;
    view->ndims = dims;
    view->base_tensor = base_tensor;
    base_tensor->ref_count++;
    view->num_slices = 0;
    view->slices = NULL;

    if (dims > 0)
    {
        int *d_shape;
        size_t *d_strides;

        view->shape = (int *)emalloc(sizeof(int) * dims);
        memcpy(view->shape, shape, sizeof(int) * dims);

        view->strides = (size_t *)emalloc(sizeof(size_t) * dims);
        memcpy(view->strides, strides, sizeof(size_t) * dims);

        cudaMemcpy(d_shape, view->shape, dims * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, view->strides, dims * sizeof(size_t), cudaMemcpyHostToDevice);

        view->d_shape = d_shape;
        view->d_strides = d_strides;
    }
    else
    {
        view->shape = NULL;
        view->strides = NULL;
        view->d_strides = NULL;
        view->d_shape = NULL;
    }

    return view;
}

tensor_t *cuda_tensor_create_sliced_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices)
{
    if (!base_tensor || !slices)
    {
        return NULL;
    }

    size_t base_strides[MAX_DIMS];
    if (base_tensor->strides)
    {
        for (int i = 0; i < base_tensor->ndims; ++i)
        {
            base_strides[i] = base_tensor->strides[i];
        }
    }
    else
    {
        size_t stride = 1;
        for (int i = base_tensor->ndims - 1; i >= 0; --i)
        {
            base_strides[i] = stride;
            stride *= (size_t)base_tensor->shape[i];
        }
    }

    size_t element_offset = base_tensor->is_view ? (base_tensor->gpu_offset / sizeof(float)) : 0;

    int view_shape[MAX_DIMS];
    size_t view_strides[MAX_DIMS];
    int view_ndims = 0;

    for (int i = 0; i < base_tensor->ndims; ++i)
    {
        slice_info_t slice = (i < num_slices) ? slices[i] : (slice_info_t){.type = SLICE_ALL};

        switch (slice.type)
        {
        case SLICE_ALL:
            view_shape[view_ndims] = base_tensor->shape[i];
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;

        case SLICE_INDEX:
        {
            int index = slice.data.index;
            if (index < 0 || index >= base_tensor->shape[i])
            {
                zend_throw_error(NULL, "Index %d out of bounds for dimension %d (size %d)",
                                 index, i, base_tensor->shape[i]);
                return NULL;
            }

            view_strides[view_ndims] = 0;
            view_shape[view_ndims] = 1;
            element_offset += index * base_strides[i];
            view_ndims++;
            break;
        }

        case SLICE_RANGE:
        {
            int start = slice.data.range.start;
            int end = slice.data.range.end;
            if (start < 0 || end < start || end >= base_tensor->shape[i])
            {
                zend_throw_error(NULL, "Range [%d:%d] out of bounds for dimension %d (size %d)",
                                 start, end, i, base_tensor->shape[i]);
                return NULL;
            }
            int len = (end - start + 1);
            element_offset += (size_t)start * base_strides[i];

            view_shape[view_ndims] = len;
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;
        }

        default:
            zend_throw_error(NULL, "Invalid slice type for dimension %d", i);
            return NULL;
        }
    }

    size_t view_total = 1;
    for (int i = 0; i < view_ndims; ++i)
        view_total *= (size_t)view_shape[i];

    size_t base_total = base_tensor->total_size;
    if (element_offset >= base_total)
    {
        zend_throw_error(NULL, "Slice offset %zu out of bounds (base size %zu)", element_offset, base_total);
        return NULL;
    }
    if (element_offset + view_total > base_total)
    {
        zend_throw_error(NULL, "Slice region (offset %zu length %zu) out of bounds (base size %zu)",
                         element_offset, view_total, base_total);
        return NULL;
    }

    tensor_t *view = cuda_tensor_create_view(base_tensor, view_shape, view_strides, view_ndims, element_offset, view_total);
    if (num_slices > 0)
    {
        view->num_slices = num_slices;
        view->slices = (slice_info_t *)emalloc(sizeof(slice_info_t) * num_slices);
        memcpy(view->slices, slices, sizeof(slice_info_t) * num_slices);
    }

    return view;
}

int cuda_tensor_set_scalar(tensor_t *tensor, size_t element_offset, float scalar_value)
{
    size_t byte_offset = element_offset * sizeof(float);

    void *gpu_destination = (char *)tensor->data + byte_offset;

    cudaError_t err = cudaMemcpy(gpu_destination, &scalar_value, sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        return FAILURE;
    }
    return SUCCESS;
}

tensor_t *cuda_tensor_create_dim_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices)
{
    if (!base_tensor || !slices)
    {
        return NULL;
    }

    size_t base_strides[MAX_DIMS];
    if (base_tensor->strides)
    {
        for (int i = 0; i < base_tensor->ndims; ++i)
        {
            base_strides[i] = base_tensor->strides[i];
        }
    }
    else
    {
        size_t stride = 1;
        for (int i = base_tensor->ndims - 1; i >= 0; --i)
        {
            base_strides[i] = stride;
            stride *= (size_t)base_tensor->shape[i];
        }
    }

    size_t element_offset = base_tensor->is_view ? (base_tensor->gpu_offset / sizeof(float)) : 0;

    int view_shape[MAX_DIMS];
    size_t view_strides[MAX_DIMS];
    int view_ndims = 0;

    for (int i = 0; i < base_tensor->ndims; ++i)
    {
        slice_info_t slice = (i < num_slices) ? slices[i] : (slice_info_t){.type = SLICE_ALL};

        switch (slice.type)
        {
        case SLICE_ALL:
            view_shape[view_ndims] = base_tensor->shape[i];
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;

        case SLICE_INDEX:
        {
            int index = slice.data.index;
            if (index < 0 || index >= base_tensor->shape[i])
            {
                zend_throw_error(NULL, "Index %d out of bounds for dimension %d (size %d)",
                                 index, i, base_tensor->shape[i]);
                return NULL;
            }

            element_offset += (size_t)index * base_strides[i];
            break;
        }

        case SLICE_RANGE:
        {
            int start = slice.data.range.start;
            int end = slice.data.range.end;
            if (start < 0 || end < start || end >= base_tensor->shape[i])
            {
                zend_throw_error(NULL, "Range [%d:%d] out of bounds for dimension %d (size %d)",
                                 start, end, i, base_tensor->shape[i]);
                return NULL;
            }
            int len = (end - start + 1);
            element_offset += (size_t)start * base_strides[i];

            view_shape[view_ndims] = len;
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;
        }

        default:
            zend_throw_error(NULL, "Invalid slice type for dimension %d", i);
            return NULL;
        }
    }

    size_t view_total = 1;
    for (int i = 0; i < view_ndims; ++i)
        view_total *= (size_t)view_shape[i];

    size_t base_total = base_tensor->total_size;
    if (element_offset >= base_total)
    {
        zend_throw_error(NULL, "Slice offset %zu out of bounds (base size %zu)", element_offset, base_total);
        return NULL;
    }
    if (element_offset + view_total > base_total)
    {
        zend_throw_error(NULL, "Slice region (offset %zu length %zu) out of bounds (base size %zu)",
                         element_offset, view_total, base_total);
        return NULL;
    }

    tensor_t *view = cuda_tensor_create_view(base_tensor, view_shape, view_strides, view_ndims, element_offset, view_total);
    if (num_slices > 0)
    {
        view->num_slices = num_slices;
        view->slices = (slice_info_t *)emalloc(sizeof(slice_info_t) * num_slices);
        memcpy(view->slices, slices, sizeof(slice_info_t) * num_slices);
    }

    return view;
}

void cuda_set_initialized(int status)
{
    cuda_is_initialized = status;
}

int cuda_initialized()
{
    return cuda_is_initialized;
}

void cuda_tensor_destroy(tensor_t *tensor)
{
    if (!tensor)
        return;

    if (tensor->is_view && !tensor->base_tensor)
    {
        if (tensor->shape)
            efree(tensor->shape);
        if (tensor->strides)
            efree(tensor->strides);
        if (tensor->slices)
            efree(tensor->slices);

        if (tensor->d_shape)
            cudaFree(tensor->d_shape);
        if (tensor->d_strides)
            cudaFree(tensor->d_strides);

        efree(tensor);
        return;
    }

    tensor->ref_count--;
    if (tensor->ref_count > 0)
        return;

    if (tensor->is_view)
    {
        if (tensor->base_tensor)
        {
            tensor->base_tensor->ref_count--;
            if (tensor->base_tensor->ref_count <= 0)
            {
                cuda_tensor_destroy(tensor->base_tensor);
            }
        }

        if (tensor->slices)
            efree(tensor->slices);
    }
    else
    {
        if (tensor->data)
        {
            tensor_mem_free(tensor->data);
            tensor->data = NULL;
        }
    }

    if (tensor->d_shape)
        cudaFree(tensor->d_shape);
    if (tensor->d_strides)
        cudaFree(tensor->d_strides);

    if (tensor->shape)
        efree(tensor->shape);
    if (tensor->strides)
        efree(tensor->strides);

    efree(tensor);
}
