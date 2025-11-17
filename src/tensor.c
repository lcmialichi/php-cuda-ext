#include "tensor.h"
#include "php.h"
#include "Zend/zend_API.h"

static int cuda_is_initialized = 0;

int tensor_init()
{
    if (cuda_initialized())
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

    cuda_set_initialized(1);
    return 1;
}

tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices)
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

    size_t byte_offset = element_offset * sizeof(float);

    tensor_t *view = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!view)
        return NULL;

    memset(view, 0, sizeof(tensor_t));

    view->is_view = 1;
    view->gpu_offset = 0;
    view->data = (float *)((char *)base_tensor->data + byte_offset);
    view->ref_count = 1;
    view->ndims = view_ndims;
    if (view_ndims > 0)
    {
        view->shape = (int *)emalloc(sizeof(int) * view_ndims);
        memcpy(view->shape, view_shape, sizeof(int) * view_ndims);

        view->strides = (size_t *)emalloc(sizeof(size_t) * view_ndims);
        memcpy(view->strides, view_strides, sizeof(size_t) * view_ndims);
    }
    else
    {
        view->shape = NULL;
        view->strides = NULL;
    }

    view->total_size = view_total;

    view->base_tensor = base_tensor;
    base_tensor->ref_count++;

    if (num_slices > 0)
    {
        view->num_slices = num_slices;
        view->slices = (slice_info_t *)emalloc(sizeof(slice_info_t) * num_slices);
        memcpy(view->slices, slices, sizeof(slice_info_t) * num_slices);
    }
    else
    {
        view->num_slices = 0;
        view->slices = NULL;
    }

    cudnnCreateTensorDescriptor(&view->desc);
    if (view_ndims > 0)
    {
        int cudnn_dims[MAX_DIMS];
        int cudnn_strides[MAX_DIMS];
        for (int i = 0; i < view_ndims; ++i)
        {
            cudnn_dims[i] = view_shape[i];
            cudnn_strides[i] = (int)view_strides[i];
        }
        cudnnSetTensorNdDescriptor(view->desc, CUDNN_DATA_FLOAT, view_ndims, cudnn_dims, cudnn_strides);
    }
    else
    {
        int dims[1] = {1};
        int strides_i[1] = {1};
        cudnnSetTensorNdDescriptor(view->desc, CUDNN_DATA_FLOAT, 1, dims, strides_i);
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
            cudaFree(tensor->data);
            tensor->data = NULL;
        }
    }

    if (tensor->shape)
        efree(tensor->shape);
    if (tensor->desc)
        cudnnDestroyTensorDescriptor(tensor->desc);

    efree(tensor);
}

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create(shape, ndims, NULL);
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
