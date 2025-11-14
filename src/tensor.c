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

    size_t base_strides[MAX_DIMS] = {0};
    size_t stride = 1;

    for (int i = base_tensor->ndims - 1; i >= 0; i--)
    {
        base_strides[i] = stride;
        stride *= base_tensor->shape[i];
    }

    size_t element_offset = 0;
    if (base_tensor->is_view)
    {
        element_offset = base_tensor->gpu_offset / sizeof(float);
    }

    int new_ndims = num_slices;
    int new_shape[MAX_DIMS];

    for (int i = 0; i < num_slices; i++)
    {
        if (i < base_tensor->ndims)
        {
            switch (slices[i].type)
            {
            case SLICE_ALL:
                new_shape[i] = base_tensor->shape[i];
                break;
            case SLICE_RANGE:
                new_shape[i] = slices[i].data.range.end - slices[i].data.range.start + 1;
                element_offset += slices[i].data.range.start * base_strides[i];
                break;
            case SLICE_INDEX:
                new_shape[i] = 1;
                element_offset += slices[i].data.index * base_strides[i];
                break;
            }
        }
        else
        {
            new_shape[i] = 1;
        }
    }

    size_t byte_offset = element_offset * sizeof(float);

    size_t base_total_elements = base_tensor->total_size;
    if (element_offset >= base_total_elements)
    {
        return NULL;
    }

    tensor_t *view = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!view)
        return NULL;

    view->is_view = 1;
    view->gpu_offset = byte_offset;
    view->data = (float *)((char *)base_tensor->data + byte_offset);
    view->ref_count = 1;
    view->total_size = 1;
    view->ndims = new_ndims;
    view->shape = (int *)emalloc(new_ndims * sizeof(int));
    memcpy(view->shape, new_shape, new_ndims * sizeof(int));

    for (int i = 0; i < new_ndims; i++)
    {
        view->total_size *= new_shape[i];
    }

    view->base_tensor = base_tensor;
    view->base_tensor->ref_count++;

    view->slices = (slice_info_t *)emalloc(num_slices * sizeof(slice_info_t));
    memcpy(view->slices, slices, num_slices * sizeof(slice_info_t));
    view->num_slices = num_slices;

    cudnnCreateTensorDescriptor(&view->desc);

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

        cudnnSetTensorNdDescriptor(view->desc, CUDNN_DATA_FLOAT, 4, dims, strides);
    }
    else
    {
        int *strides = (int *)emalloc(new_ndims * sizeof(int));
        strides[new_ndims - 1] = 1;
        for (int i = new_ndims - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * new_shape[i + 1];
        }
        cudnnSetTensorNdDescriptor(view->desc, CUDNN_DATA_FLOAT, new_ndims, new_shape, strides);
        efree(strides);
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

    if (tensor->is_view && !tensor->base_tensor) {
        zend_error(E_WARNING, "Attempting to destroy a view with no base tensor!");
        return;
    }

    if (!tensor->is_view && tensor->ref_count > 1) {
        zend_error(E_WARNING, "Attempting to destroy a base tensor that still has active views!");
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
    {
        return NULL;
    }

    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
        return NULL;

    tensor->ndims = ndims;
    tensor->shape = (int *)emalloc(ndims * sizeof(int));
    tensor->is_view = 0;
    memcpy(tensor->shape, shape, ndims * sizeof(int));

    tensor->total_size = 1;
    tensor->ref_count = 1;

    for (int i = 0; i < ndims; i++)
    {
        tensor->total_size *= shape[i];
    }

    cudaMalloc((void **)&tensor->data, tensor->total_size * sizeof(float));
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
