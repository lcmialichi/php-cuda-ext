#include <cuda_runtime.h>
#include <math.h>
#include "cuda_kernels.h"
#include <float.h>
#include <cstdio>

#define SUCCESS 0
#define FAILURE 1
struct MatMulParams {
    const float* A;
    const float* B;
    float* C;

    int batch;
    int M;
    int K;
    int N;

    size_t strideA;
    size_t strideB;
    size_t strideC;
};

extern "C"
{
    __global__ void fill_kernel(float *data, float value, size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            data[idx] = value;
    }

    __global__ void scale_kernel(float *data, size_t size, float min_value, float max_value)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size)
        {
            float raw_rand = data[idx];
            float range = max_value - min_value;

            data[idx] = min_value + range * raw_rand;
        }
    }

    __global__ void clip_kernel(float *a, float min_val, float max_val, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = fminf(fmaxf(a[i], min_val), max_val);
        }
    }

    __global__ void relu_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = fmaxf(a[i], 0.0f);
        }
    }

    __global__ void sigmoid_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = 1.0f / (1.0f + expf(-a[i]));
        }
    }

    __global__ void tanh_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = tanhf(a[i]);
        }
    }

    __global__ void concat_kernel_int32(ConcatParams *params, int *output)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= params->outer_dims * params->output_axis_size * params->inner_dims)
        {
            return;
        }

        size_t outer_idx = idx / (params->output_axis_size * params->inner_dims);
        size_t axis_idx = (idx / params->inner_dims) % params->output_axis_size;
        size_t inner_idx = idx % params->inner_dims;

        int tensor_idx = 0;
        size_t current_offset = 0;

        for (tensor_idx = 0; tensor_idx < params->num_tensors; tensor_idx++)
        {
            size_t tensor_axis_size = params->input_axis_sizes[tensor_idx];
            if (axis_idx < current_offset + tensor_axis_size)
            {
                break;
            }
            current_offset += tensor_axis_size;
        }

        if (tensor_idx >= params->num_tensors)
        {
            return;
        }

        size_t local_axis_idx = axis_idx - current_offset;
        int *input_tensor = (int *)params->input_ptrs[tensor_idx];

        size_t input_offset = outer_idx * params->input_axis_sizes[tensor_idx] * params->inner_dims;
        input_offset += local_axis_idx * params->inner_dims;
        input_offset += inner_idx;

        size_t output_offset = outer_idx * params->output_axis_size * params->inner_dims;
        output_offset += axis_idx * params->inner_dims;
        output_offset += inner_idx;

        output[output_offset] = input_tensor[input_offset];
    }

    __global__ void concat_kernel_float(ConcatParams *params, float *output)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= params->outer_dims * params->output_axis_size * params->inner_dims)
        {
            return;
        }

        size_t outer_idx = idx / (params->output_axis_size * params->inner_dims);
        size_t axis_idx = (idx / params->inner_dims) % params->output_axis_size;
        size_t inner_idx = idx % params->inner_dims;

        int tensor_idx = 0;
        size_t current_offset = 0;

        for (tensor_idx = 0; tensor_idx < params->num_tensors; tensor_idx++)
        {
            size_t tensor_axis_size = params->input_axis_sizes[tensor_idx];
            if (axis_idx < current_offset + tensor_axis_size)
            {
                break;
            }
            current_offset += tensor_axis_size;
        }

        if (tensor_idx >= params->num_tensors)
        {
            return;
        }

        size_t local_axis_idx = axis_idx - current_offset;
        float *input_tensor = (float *)params->input_ptrs[tensor_idx];

        size_t input_offset = outer_idx * params->input_axis_sizes[tensor_idx] * params->inner_dims;
        input_offset += local_axis_idx * params->inner_dims;
        input_offset += inner_idx;

        size_t output_offset = outer_idx * params->output_axis_size * params->inner_dims;
        output_offset += axis_idx * params->inner_dims;
        output_offset += inner_idx;

        output[output_offset] = input_tensor[input_offset];
    }

    void get_grid_config(size_t size, int *grid_size, int *block_size)
    {
        *block_size = 256;
        *grid_size = (size + *block_size - 1) / *block_size;
    }

    int launch_scale_kernel_host(float *data, size_t size, float min_value, float max_value)
    {
        int grid_size, block_size;
        get_grid_config(size, &grid_size, &block_size);

        scale_kernel<<<grid_size, block_size>>>(data, size, min_value, max_value);

        if (cudaPeekAtLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        {
            return FAILURE;
        }

        return SUCCESS;
    }

    void launch_fill_kernel(float *data, float value, size_t size)
    {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        fill_kernel<<<blocks, threads>>>(data, value, size);
    }

    int launch_concat_kernel_host(
        tensor_t **input_tensors,
        int num_tensors,
        tensor_t *output_tensor,
        int axis,
        size_t *input_axis_offsets,
        size_t *input_strides_axis,
        size_t output_stride_axis,
        size_t outer_dims,
        size_t inner_dims,
        int output_axis_size)
    {
        size_t total_elements = outer_dims * output_axis_size * inner_dims;
        int grid_size, block_size;
        get_grid_config(total_elements, &grid_size, &block_size);

        ConcatParams params;
        params.num_tensors = num_tensors;
        params.outer_dims = outer_dims;
        params.inner_dims = inner_dims;
        params.output_stride = output_stride_axis;
        params.output_axis_size = output_axis_size;

        for (int i = 0; i < num_tensors; i++)
        {
            params.input_ptrs[i] = input_tensors[i]->data;
            params.input_axis_sizes[i] = input_tensors[i]->shape[axis];
            params.input_axis_offsets[i] = input_axis_offsets[i];
            params.input_strides_axis[i] = input_strides_axis[i];
        }

        ConcatParams *d_params;
        cudaError_t status = cudaMalloc(&d_params, sizeof(ConcatParams));
        if (status != cudaSuccess)
        {
            return FAILURE;
        }

        status = cudaMemcpy(d_params, &params, sizeof(ConcatParams), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
        {
            cudaFree(d_params);
            return FAILURE;
        }

        switch (output_tensor->dtype)
        {
        case DTYPE_FLOAT:
            concat_kernel_float<<<grid_size, block_size>>>(
                d_params,
                (float *)output_tensor->data);
            break;
        case DTYPE_INT:
            concat_kernel_int32<<<grid_size, block_size>>>(
                d_params,
                (int *)output_tensor->data);
            break;
        default:
            cudaFree(d_params);
            return FAILURE;
        }

        status = cudaPeekAtLastError();
        if (status != cudaSuccess)
        {
            cudaFree(d_params);
            return FAILURE;
        }

        status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
        {
            cudaFree(d_params);
            return FAILURE;
        }

        cudaFree(d_params);
        return SUCCESS;
    }

    __device__ int compute_batch_index(int batch_idx, int *final_shape, int *broadcast_flags,
                                       int max_ndims, int tensor_ndims)
    {
        int result = 0;
        int stride = 1;

        for (int i = max_ndims - 3; i >= 0; i--)
        {
            int dim_size = final_shape[i];
            int coord = (batch_idx / stride) % dim_size;

            if (!broadcast_flags[i])
            {
                result += coord * stride;
            }

            stride *= dim_size;
        }

        return result;
    }

    __global__ void matmul_kernel(float *a, float *b, float *c,
                                  int m, int n, int k,
                                  size_t a_stride0, size_t a_stride1,
                                  size_t b_stride0, size_t b_stride1,
                                  size_t c_stride0, size_t c_stride1)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < k)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                size_t a_idx = row * a_stride0 + i * a_stride1;
                size_t b_idx = i * b_stride0 + col * b_stride1;
                sum += a[a_idx] * b[b_idx];
            }

            size_t c_idx = row * c_stride0 + col * c_stride1;
            c[c_idx] = sum;
        }
    }

    __global__ void matmul_batched_kernel(MatMulParams p)
    {
        int batch_id = blockIdx.z;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_id >= p.batch || row >= p.M || col >= p.N)
            return;

        const float *A = p.A + batch_id * p.strideA;
        const float *B = p.B + batch_id * p.strideB;
        float *C = p.C + batch_id * p.strideC;

        float sum = 0.0f;

        for (int k = 0; k < p.K; k++)
            sum += A[row * p.K + k] * B[k * p.N + col];

        C[row * p.N + col] = sum;
    }

    int cuda_batched_matmul_launcher(
        float *a, float *b, float *c,
        int *a_shape, size_t *a_strides,
        int *b_shape, size_t *b_strides,
        int *c_shape, size_t *c_strides,
        int a_ndims, int b_ndims)
    {
        MatMulParams p;

        p.A = a;
        p.B = b;
        p.C = c;

        p.batch = a_shape[0];
        p.M = a_shape[1];
        p.K = a_shape[2];
        p.N = b_shape[2];

        p.strideA = a_strides[0];
        p.strideB = b_strides[0];
        p.strideC = c_strides[0];

        // Grid / block
        dim3 block(16, 16);
        dim3 grid(
            (p.N + block.x - 1) / block.x,
            (p.M + block.y - 1) / block.y,
            p.batch);

        // Lançar kernel
        matmul_batched_kernel<<<grid, block>>>(p);

        return cudaGetLastError() == cudaSuccess ? 1 : 0;
    }

    int cuda_matmul_launcher(float *a, float *b, float *c,
                             int m, int n, int k,
                             size_t a_stride0, size_t a_stride1,
                             size_t b_stride0, size_t b_stride1,
                             size_t c_stride0, size_t c_stride1)
    {
        dim3 blocks((k + 31) / 32, (m + 31) / 32);
        dim3 threads(32, 32);

        matmul_kernel<<<blocks, threads>>>(a, b, c, m, n, k,
                                           a_stride0, a_stride1,
                                           b_stride0, b_stride1,
                                           c_stride0, c_stride1);

        return cudaGetLastError() == cudaSuccess ? 1 : 0;
    }

    void launch_clip_kernel(float *a, float min_val, float max_val, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        clip_kernel<<<blocks, threads>>>(a, min_val, max_val, result, n);
    }

    void launch_relu_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_sigmoid_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        sigmoid_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_tanh_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        tanh_kernel<<<blocks, threads>>>(a, result, n);
    }
}