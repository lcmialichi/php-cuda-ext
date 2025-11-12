#include <cuda_runtime.h>
#include <math.h>
#include "cuda_kernels.h"
#include <float.h>
#include <cstdio>

extern "C"
{
    __global__ void broadcast_kernel(float *a, float *b, float *result,
                                     int *a_strides, int a_dims,
                                     int *b_strides, int b_dims,
                                     int *result_shape, int result_dims,
                                     size_t total_elements, int operation_type)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements)
            return;

        int temp_idx = idx;
        int a_idx = 0, b_idx = 0;

        for (int i = result_dims - 1; i >= 0; i--)
        {
            int coord = temp_idx % result_shape[i];
            temp_idx /= result_shape[i];

            if (i >= result_dims - a_dims)
            {
                int a_dim_idx = i - (result_dims - a_dims);
                a_idx += coord * a_strides[a_dim_idx];
            }

            if (i >= result_dims - b_dims)
            {
                int b_dim_idx = i - (result_dims - b_dims);
                b_idx += coord * b_strides[b_dim_idx];
            }
        }

        switch (operation_type)
        {
        case OP_ADD:
            result[idx] = a[a_idx] + b[b_idx];
            break;
        case OP_SUB:
            result[idx] = a[a_idx] - b[b_idx];
            break;
        case OP_MUL:
            result[idx] = a[a_idx] * b[b_idx];
            break;
        case OP_DIV:
            result[idx] = a[a_idx] / b[b_idx];
            break;
        case OP_POW:
            result[idx] = powf(a[a_idx], b[b_idx]);
            break;
        case OP_GT:
            result[idx] = (a[a_idx] > b[b_idx]) ? 1.0f : 0.0f;
            break;
        case OP_LT:
            result[idx] = (a[a_idx] < b[b_idx]) ? 1.0f : 0.0f;
            break;
        case OP_EQ:
            result[idx] = (fabsf(a[a_idx] - b[b_idx]) < 1e-6f) ? 1.0f : 0.0f;
            break;
        case OP_NE:
            result[idx] = (fabsf(a[a_idx] - b[b_idx]) >= 1e-6f) ? 1.0f : 0.0f;
            break;
        case OP_GE:
            result[idx] = (a[a_idx] >= b[b_idx]) ? 1.0f : 0.0f;
            break;
        case OP_LE:
            result[idx] = (a[a_idx] <= b[b_idx]) ? 1.0f : 0.0f;
            break;
        }
    }

    __global__ void add_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] + b[i];
    }

    __global__ void subtract_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] - b[i];
    }

    __global__ void multiply_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] * b[i];
    }

    __global__ void divide_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] / b[i];
    }

    __global__ void power_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = powf(a[i], b[i]);
    }

    __global__ void scalar_add_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] + scalar;
    }

    __global__ void scalar_subtract_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] - scalar;
    }

    __global__ void scalar_multiply_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] * scalar;
    }

    __global__ void scalar_divide_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = a[i] / scalar;
    }

    __global__ void scalar_power_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = powf(a[i], scalar);
    }

    __global__ void sqrt_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = sqrtf(a[i]);
    }

    __global__ void exp_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = expf(a[i]);
    }

    __global__ void log_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = logf(a[i]);
    }

    __global__ void sin_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = sinf(a[i]);
    }

    __global__ void cos_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = cosf(a[i]);
    }

    __global__ void tan_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = tanf(a[i]);
    }

    __global__ void abs_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = fabsf(a[i]);
    }

    __global__ void negate_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = -a[i];
    }

    __global__ void reciprocal_kernel(float *a, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = 1.0f / a[i];
    }

    __global__ void greater_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }

    __global__ void less_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
    }

    __global__ void equal_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
    }

    __global__ void greater_equal_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
    }

    __global__ void less_equal_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
    }

    __global__ void not_equal_kernel(float *a, float *b, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            result[i] = (a[i] != b[i]) ? 1.0f : 0.0f;
    }

    __global__ void scalar_greater_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = (a[i] > scalar) ? 1.0f : 0.0f;
        }
    }

    __global__ void scalar_less_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = (a[i] < scalar) ? 1.0f : 0.0f;
        }
    }

    __global__ void scalar_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = (fabsf(a[i] - scalar) < 1e-6f) ? 1.0f : 0.0f;
        }
    }

    __global__ void scalar_not_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = (fabsf(a[i] - scalar) >= 1e-6f) ? 1.0f : 0.0f;
        }
    }

    __global__ void scalar_greater_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = (a[i] >= scalar) ? 1.0f : 0.0f;
        }
    }

    __global__ void scalar_less_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            result[i] = (a[i] <= scalar) ? 1.0f : 0.0f;
        }
    }

    __global__ void sum_kernel(float *a, float *result, int n)
    {
        extern __shared__ float shared[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        shared[tid] = (i < n) ? a[i] : 0.0f;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            atomicAdd(result, shared[0]);
        }
    }

    __device__ float atomicMaxFloat(float *address, float val)
    {
        int *address_as_int = (int *)address;
        int old = *address_as_int, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    __device__ float atomicMinFloat(float *address, float val)
    {
        int *address_as_int = (int *)address;
        int old = *address_as_int, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __float_as_int(fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    __global__ void max_kernel(float *a, float *result, int n)
    {
        extern __shared__ float shared[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        shared[tid] = (i < n) ? a[i] : -1e30f;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared[tid] = fmaxf(shared[tid], shared[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            atomicMaxFloat(result, shared[0]);
        }
    }

    __global__ void min_kernel(float *a, float *result, int n)
    {
        extern __shared__ float shared[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        shared[tid] = (i < n) ? a[i] : FLT_MAX;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared[tid] = fminf(shared[tid], shared[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            atomicMinFloat(result, __float_as_int(shared[0]));
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
    
    void launch_broadcast_kernel(float *a, float *b, float *result,
                                 int *a_strides, int a_dims,
                                 int *b_strides, int b_dims,
                                 int *result_shape, int result_dims,
                                 size_t total_elements, int operation_type)
    {

        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        int *d_a_strides, *d_b_strides, *d_result_shape;

        cudaMalloc(&d_a_strides, a_dims * sizeof(int));
        cudaMalloc(&d_b_strides, b_dims * sizeof(int));
        cudaMalloc(&d_result_shape, result_dims * sizeof(int));

        cudaMemcpy(d_a_strides, a_strides, a_dims * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_strides, b_strides, b_dims * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_result_shape, result_shape, result_dims * sizeof(int), cudaMemcpyHostToDevice);

        broadcast_kernel<<<blocks, threads>>>(
            a, b, result,
            d_a_strides, a_dims,
            d_b_strides, b_dims,
            d_result_shape, result_dims,
            total_elements, operation_type);

        cudaFree(d_a_strides);
        cudaFree(d_b_strides);
        cudaFree(d_result_shape);
    }

    void launch_add_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_subtract_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        subtract_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_multiply_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        multiply_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_divide_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        divide_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_power_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        power_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_scalar_add_kernel(float *a, float scalar, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scalar_add_kernel<<<blocks, threads>>>(a, scalar, result, n);
    }

    void launch_scalar_subtract_kernel(float *a, float scalar, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scalar_subtract_kernel<<<blocks, threads>>>(a, scalar, result, n);
    }

    void launch_scalar_multiply_kernel(float *a, float scalar, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scalar_multiply_kernel<<<blocks, threads>>>(a, scalar, result, n);
    }

    void launch_scalar_divide_kernel(float *a, float scalar, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scalar_divide_kernel<<<blocks, threads>>>(a, scalar, result, n);
    }

    void launch_scalar_power_kernel(float *a, float scalar, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scalar_power_kernel<<<blocks, threads>>>(a, scalar, result, n);
    }

    void launch_sqrt_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        sqrt_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_exp_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        exp_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_log_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        log_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_sin_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        sin_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_cos_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        cos_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_tan_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        tan_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_abs_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        abs_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_negate_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        negate_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_reciprocal_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        reciprocal_kernel<<<blocks, threads>>>(a, result, n);
    }

    void launch_greater_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        greater_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_less_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        less_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_equal_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        equal_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_greater_equal_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        greater_equal_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_less_equal_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        less_equal_kernel<<<blocks, threads>>>(a, b, result, n);
    }

    void launch_not_equal_kernel(float *a, float *b, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        not_equal_kernel<<<blocks, threads>>>(a, b, result, n);
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

    void launch_sum_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        size_t shared_mem = threads * sizeof(float);

        cudaMemset(result, 0, sizeof(float));
        sum_kernel<<<blocks, threads, shared_mem>>>(a, result, n);
    }

    void launch_max_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        size_t shared_mem = threads * sizeof(float);

        float init = -FLT_MAX;
        cudaMemcpy(result, &init, sizeof(float), cudaMemcpyHostToDevice);
        max_kernel<<<blocks, threads, shared_mem>>>(a, result, n);
    }

    void launch_min_kernel(float *a, float *result, int n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        size_t shared_mem = threads * sizeof(float);

        float init = FLT_MAX;
        cudaMemcpy(result, &init, sizeof(float), cudaMemcpyHostToDevice);
        min_kernel<<<blocks, threads, shared_mem>>>(a, result, n);
    }

    void launch_scalar_greater_kernel(float *a, float scalar, float *result, int n)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_greater_kernel<<<numBlocks, blockSize>>>(a, scalar, result, n);
    }

    void launch_scalar_less_kernel(float *a, float scalar, float *result, int n)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_less_kernel<<<numBlocks, blockSize>>>(a, scalar, result, n);
    }

    void launch_scalar_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_equal_kernel<<<numBlocks, blockSize>>>(a, scalar, result, n);
    }

    void launch_scalar_not_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_not_equal_kernel<<<numBlocks, blockSize>>>(a, scalar, result, n);
    }

    void launch_scalar_greater_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_greater_equal_kernel<<<numBlocks, blockSize>>>(a, scalar, result, n);
    }

    void launch_scalar_less_equal_kernel(float *a, float scalar, float *result, int n)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        scalar_less_equal_kernel<<<numBlocks, blockSize>>>(a, scalar, result, n);
    }
}