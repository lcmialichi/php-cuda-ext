#include "tensor.h"
#include "php.h"

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

    tensor->ref_count--;

    if (tensor->ref_count <= 0) {
        if (tensor->data) {
            cudaFree(tensor->data);
            tensor->data = NULL;
        }
        if (tensor->shape) {
            efree(tensor->shape);
            tensor->shape = NULL;
        }
        if (tensor->desc) {
            cudnnDestroyTensorDescriptor(tensor->desc);
            tensor->desc = NULL;
        }
        efree(tensor);
    } else {
        printf("TENSOR STILL IN USE: data=%p, ref_count=%d\n", tensor->data, tensor->ref_count);
    }
}

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create(shape, ndims, NULL);
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
