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

int calculate_view_shape_and_offset(tensor_t *base_tensor, slice_info_t *slices, int num_slices,
                                   int *new_shape, int *new_ndims, size_t *gpu_offset)
{
    *new_ndims = num_slices;
    *gpu_offset = 0;
    
    printf("=== CALCULATE_VIEW_SHAPE_AND_OFFSET ===\n");
    printf("Base tensor shape: ");
    for (int i = 0; i < base_tensor->ndims; i++) {
        printf("%d ", base_tensor->shape[i]);
    }
    printf("(ndims=%d)\n", base_tensor->ndims);
    
    size_t strides[MAX_DIMS] = {0};
    size_t stride = 1;
    for (int i = base_tensor->ndims - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= base_tensor->shape[i];
        printf("Stride[%d] = %zu\n", i, strides[i]);
    }
    
    for (int i = 0; i < num_slices; i++) {
        printf("Slice[%d]: type=%d, ", i, slices[i].type);
        
        if (i < base_tensor->ndims) {
            switch (slices[i].type) {
                case SLICE_ALL:
                    new_shape[i] = base_tensor->shape[i];
                    printf("ALL -> shape=%d, offset+=0\n", new_shape[i]);
                    break;
                    
                case SLICE_RANGE:
                    printf("RANGE(%d,%d) -> ", slices[i].data.range.start, slices[i].data.range.end);
                    if (slices[i].data.range.start < 0 || 
                        slices[i].data.range.end >= base_tensor->shape[i] ||
                        slices[i].data.range.start > slices[i].data.range.end) {
                        printf("INVALID RANGE!\n");
                        return 0;
                    }
                    new_shape[i] = slices[i].data.range.end - slices[i].data.range.start + 1;
                    *gpu_offset += slices[i].data.range.start * strides[i];
                    printf("shape=%d, offset+=%d*%zu=%zu, total_offset=%zu\n", 
                           new_shape[i], slices[i].data.range.start, strides[i], 
                           slices[i].data.range.start * strides[i], *gpu_offset);
                    break;
                    
                case SLICE_INDEX:
                    printf("INDEX(%d) -> ", slices[i].data.index);
                    if (slices[i].data.index < 0 || slices[i].data.index >= base_tensor->shape[i]) {
                        printf("INVALID INDEX!\n");
                        return 0;
                    }
                    new_shape[i] = 1;
                    *gpu_offset += slices[i].data.index * strides[i];
                    printf("shape=%d, offset+=%d*%zu=%zu, total_offset=%zu\n", 
                           new_shape[i], slices[i].data.index, strides[i],
                           slices[i].data.index * strides[i], *gpu_offset);
                    break;
            }
        } else {
            new_shape[i] = 1;
            printf("EXTRA -> shape=1\n");
        }
    }
    
    *gpu_offset *= sizeof(float);
    printf("FINAL OFFSET: %zu bytes (%zu elements)\n", *gpu_offset, *gpu_offset / sizeof(float));
    printf("====================================\n");
    
    return 1;
}

tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices)
{
    if (!base_tensor || !slices) {
        return NULL;
    }
    
    size_t base_strides[MAX_DIMS] = {0};
    size_t stride = 1;
    
    for (int i = base_tensor->ndims - 1; i >= 0; i--) {
        base_strides[i] = stride;
        stride *= base_tensor->shape[i];
    }
    
    size_t element_offset = 0;
    if (base_tensor->is_view) {
        element_offset = base_tensor->gpu_offset / sizeof(float);
    }
    
    int new_ndims = num_slices;
    int new_shape[MAX_DIMS];
    
    for (int i = 0; i < num_slices; i++) {
        if (i < base_tensor->ndims) {
            switch (slices[i].type) {
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
        } else {
            new_shape[i] = 1;
        }
    }
    
    size_t byte_offset = element_offset * sizeof(float);
    
    size_t base_total_elements = base_tensor->total_size;
    if (element_offset >= base_total_elements) {
        return NULL;
    }
    
    tensor_t *view = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!view) return NULL;
    
    view->is_view = 1;
    view->gpu_offset = byte_offset;
    view->data = (float*)((char*)base_tensor->data + byte_offset);
    
    view->ndims = new_ndims;
    view->shape = (int *)emalloc(new_ndims * sizeof(int));
    memcpy(view->shape, new_shape, new_ndims * sizeof(int));
    
    view->total_size = 1;
    for (int i = 0; i < new_ndims; i++) {
        view->total_size *= new_shape[i];
    }
    
    view->ref_count = 1;
    
    if (base_tensor->is_view) {
        view->base_tensor = base_tensor->base_tensor;
    } else {
        view->base_tensor = base_tensor;
    }
    
    view->base_tensor->ref_count++;
    
    view->slices = (slice_info_t *)emalloc(num_slices * sizeof(slice_info_t));
    memcpy(view->slices, slices, num_slices * sizeof(slice_info_t));
    view->num_slices = num_slices;
    
    cudnnCreateTensorDescriptor(&view->desc);
    
    if (new_ndims <= 4) {
        int dims[4] = {1, 1, 1, 1};
        int strides[4] = {1, 1, 1, 1};
        
        for (int i = 0; i < new_ndims && i < 4; i++) {
            dims[i] = new_shape[i];
        }
        
        strides[3] = 1;
        if (new_ndims >= 4) strides[2] = dims[3];
        if (new_ndims >= 3) strides[1] = dims[2] * strides[2];
        if (new_ndims >= 2) strides[0] = dims[1] * strides[1];
        
        cudnnSetTensorNdDescriptor(view->desc, CUDNN_DATA_FLOAT, 4, dims, strides);
    } else {
        int *strides = (int *)emalloc(new_ndims * sizeof(int));
        strides[new_ndims - 1] = 1;
        for (int i = new_ndims - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * new_shape[i + 1];
        }
        cudnnSetTensorNdDescriptor(view->desc, CUDNN_DATA_FLOAT, new_ndims, new_shape, strides);
        efree(strides);
    }
    
    printf("<<< View created: data=%p (points to base data: %p)\n", view->data, base_tensor->data);
    
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
    printf(">>> init tensor_destroy: %s at %p, ref_count now: %d\n", 
           tensor->is_view ? "VIEW" : "BASE", tensor, tensor->ref_count);

    if (!tensor) return;
    tensor->ref_count--;
    
    if (tensor->ref_count <= 0) {
        printf(">>> 💀 DESTROYING: %s at %p\n", 
               tensor->is_view ? "VIEW" : "BASE", tensor);
        
        if (tensor->is_view) {
            // ✅ CORREÇÃO: Apenas decrementar o base_tensor, NÃO destruir diretamente
            if (tensor->base_tensor) {
                printf(">>> Decrementing base_tensor %p ref_count from %d to %d\n",
                       tensor->base_tensor, tensor->base_tensor->ref_count,
                       tensor->base_tensor->ref_count - 1);
                
                tensor->base_tensor->ref_count--;
                
                // ✅ SÓ destruir o base_tensor se o ref_count dele chegar a 0
                if (tensor->base_tensor->ref_count <= 0) {
                    printf(">>> Base tensor ref_count reached 0, destroying it\n");
                    cuda_tensor_destroy(tensor->base_tensor);
                }
            }
            
            if (tensor->slices) efree(tensor->slices);
            
        } else {
            if (tensor->data) {
                cudaFree(tensor->data);
                tensor->data = NULL;
            }
        }
        
        if (tensor->shape) efree(tensor->shape);
        if (tensor->desc) cudnnDestroyTensorDescriptor(tensor->desc);
        efree(tensor);
    } else {
        printf(">>> 🔄 KEEPING: %s at %p, ref_count now: %d\n",
               tensor->is_view ? "VIEW" : "BASE", tensor, tensor->ref_count);
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
