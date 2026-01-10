#include <cuda_runtime.h>
#include <math.h>
#include "cuda_kernels.h"
#include <float.h>
#include <cstdio>
#include "../data_types.h"

#define SUCCESS 0
#define FAILURE 1
#define TILE_SIZE 32

typedef struct
{
    float *A, *B, *C;
    int *shapeA, *shapeB, *shapeC;
    size_t *strideA, *strideB, *strideC;
    int ndA, ndB, ndC;
    int M, N, K;
    int total_batches;
} MatMulParamsND;

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
        case DTYPE_FLOAT32:
            concat_kernel_float<<<grid_size, block_size>>>(
                d_params,
                (float *)output_tensor->data);
            break;
        case DTYPE_INT32:
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

    __device__ __forceinline__ size_t index_from_strides(
        const int *shape, const size_t *strides,
        int ndims, size_t linear_idx)
    {
        size_t offset = 0;

        for (int i = 0; i < ndims; i++)
        {
            size_t idx = linear_idx % shape[i];
            offset += idx * strides[i];
            linear_idx /= shape[i];
        }

        return offset;
    }

    __device__ size_t linear_index(int *coords, size_t *strides, int ndims)
    {
        size_t idx = 0;
        for (int i = 0; i < ndims; i++)
        {
            idx += coords[i] * strides[i];
        }
        return idx;
    }

    __global__ void matmul_nd_tiled_kernel(MatMulParamsND p)
    {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int batch_id = blockIdx.z;
        int global_row = blockIdx.y * TILE_SIZE + ty;
        int global_col = blockIdx.x * TILE_SIZE + tx;

        if (global_row >= p.M || global_col >= p.N || batch_id >= p.total_batches)
        {
            return;
        }

        int coordsC[MAX_DIMS] = {0};
        int batch_dims = p.ndC - 2;
        int tmp = batch_id;

        for (int i = batch_dims - 1; i >= 0; i--)
        {
            coordsC[i] = tmp % p.shapeC[i];
            tmp /= p.shapeC[i];
        }

        coordsC[p.ndC - 2] = global_row;
        coordsC[p.ndC - 1] = global_col;
        size_t idxC = 0;
        for (int i = 0; i < p.ndC; i++)
            idxC += (size_t)coordsC[i] * p.strideC[i];

        float sum = 0.0f;

        for (int k_offset = 0; k_offset < p.K; k_offset += TILE_SIZE)
        {

            int global_idx_A_load_row = global_row;
            int global_idx_A_load_col = k_offset + tx;

            int global_idx_B_load_row = k_offset + ty;
            int global_idx_B_load_col = global_col;

            size_t idxA_load = 0;
            for (int i = 0; i < batch_dims; i++)
            {
                int current_batch_coord = coordsC[i];
                if (p.ndA > i + 2 && p.shapeA[i] > 1)
                {
                    idxA_load += (size_t)current_batch_coord * p.strideA[i];
                }
            }
            if (p.ndA >= 2)
                idxA_load += (size_t)global_idx_A_load_row * p.strideA[p.ndA - 2];
            if (p.ndA >= 2)
                idxA_load += (size_t)global_idx_A_load_col * p.strideA[p.ndA - 1];

            size_t idxB_load = 0;
            for (int i = 0; i < batch_dims; i++)
            {
                int current_batch_coord = coordsC[i];
                if (p.ndB > i + 2 && p.shapeB[i] > 1)
                {
                    idxB_load += (size_t)current_batch_coord * p.strideB[i];
                }
            }
            if (p.ndB >= 2)
                idxB_load += (size_t)global_idx_B_load_row * p.strideB[p.ndB - 2];
            if (p.ndB >= 2)
                idxB_load += (size_t)global_idx_B_load_col * p.strideB[p.ndB - 1];

            if (global_idx_A_load_row < p.M && global_idx_A_load_col < p.K)
                As[ty][tx] = p.A[idxA_load];
            else
                As[ty][tx] = 0.0f;

            if (global_idx_B_load_row < p.K && global_idx_B_load_col < p.N)
                Bs[ty][tx] = p.B[idxB_load];
            else
                Bs[ty][tx] = 0.0f;

            __syncthreads(); 

            for (int k = 0; k < TILE_SIZE; ++k)
            {
                sum += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }

        p.C[idxC] = sum;
    }

    int cuda_batched_matmul_nd_launcher(
        float *A, float *B, float *C,
        int *d_shapeA, size_t *d_strideA, int ndA,
        int *d_shapeB, size_t *d_strideB, int ndB,
        int *d_shapeC, size_t *d_strideC, int ndC)
    {
        if (ndC < 2 || ndA < 2 || ndB < 2)
        {
            return 0;
        }

        int h_shapeA[MAX_DIMS], h_shapeB[MAX_DIMS], h_shapeC[MAX_DIMS];

        cudaMemcpy(h_shapeA, d_shapeA, ndA * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_shapeB, d_shapeB, ndB * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_shapeC, d_shapeC, ndC * sizeof(int), cudaMemcpyDeviceToHost);

        if (h_shapeA[ndA - 1] != h_shapeB[ndB - 2])
        {
            return 0;
        }

        int M = h_shapeA[ndA - 2];
        int N = h_shapeB[ndB - 1];
        int K = h_shapeA[ndA - 1];

        if (h_shapeC[ndC - 2] != M || h_shapeC[ndC - 1] != N)
        {
            return 0;
        }

        int total_batches = 1;
        for (int i = 0; i < ndC - 2; i++)
        {
            total_batches *= h_shapeC[i];
        }

        MatMulParamsND p_host;
        p_host.A = A;
        p_host.B = B;
        p_host.C = C;
        p_host.ndA = ndA;
        p_host.ndB = ndB;
        p_host.ndC = ndC;
        p_host.M = M;
        p_host.N = N;
        p_host.K = K;
        p_host.total_batches = total_batches;
        p_host.shapeA = d_shapeA;
        p_host.strideA = d_strideA;
        p_host.shapeB = d_shapeB;
        p_host.strideB = d_strideB;
        p_host.shapeC = d_shapeC;
        p_host.strideC = d_strideC;

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, total_batches);

        matmul_nd_tiled_kernel<<<grid, block>>>(p_host);
        cudaError_t err = cudaGetLastError();

        return err == cudaSuccess;
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