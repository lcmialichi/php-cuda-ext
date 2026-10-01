#include "matmul_kernels.h"
#include "launch_config.cuh"
#include "../tensor.h"
#include <cuda_runtime.h>
#include <string.h>

#define TILE_SIZE 32

struct MatMulParamsND
{
    float *A, *B, *C;
    int shapeA[MAX_DIMS], shapeB[MAX_DIMS], shapeC[MAX_DIMS];
    size_t strideA[MAX_DIMS], strideB[MAX_DIMS], strideC[MAX_DIMS];
    int ndA, ndB, ndC;
    int M, N, K;
    int total_batches;
};

static __global__ void matmul_kernel(float *a, float *b, float *c,
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

static __global__ void matmul_nd_tiled_kernel(MatMulParamsND params)
{
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int batch_id = blockIdx.z;
    int global_row = blockIdx.y * TILE_SIZE + ty;
    int global_col = blockIdx.x * TILE_SIZE + tx;

    if (batch_id >= params.total_batches)
        return;

    size_t batch_offset_a = 0;
    size_t batch_offset_b = 0;
    size_t batch_offset_c = 0;
    int batch_dims = params.ndC - 2;
    int remaining = batch_id;

    for (int i = batch_dims - 1; i >= 0; i--)
    {
        int coord = remaining % params.shapeC[i];
        remaining /= params.shapeC[i];
        batch_offset_c += (size_t)coord * params.strideC[i];
        if (params.ndA > i + 2 && params.shapeA[i] > 1)
            batch_offset_a += (size_t)coord * params.strideA[i];
        if (params.ndB > i + 2 && params.shapeB[i] > 1)
            batch_offset_b += (size_t)coord * params.strideB[i];
    }

    float sum = 0.0f;
    int stride_a_row = params.strideA[params.ndA - 2];
    int stride_a_col = params.strideA[params.ndA - 1];
    int stride_b_row = params.strideB[params.ndB - 2];
    int stride_b_col = params.strideB[params.ndB - 1];

    for (int k_offset = 0; k_offset < params.K; k_offset += TILE_SIZE)
    {
        int a_col = k_offset + tx;
        int b_row = k_offset + ty;

        tile_a[ty][tx] = (global_row < params.M && a_col < params.K)
            ? params.A[batch_offset_a + global_row * stride_a_row + a_col * stride_a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < params.K && global_col < params.N)
            ? params.B[batch_offset_b + b_row * stride_b_row + global_col * stride_b_col] : 0.0f;

        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
            sum += tile_a[ty][i] * tile_b[i][tx];
        __syncthreads();
    }

    if (global_row < params.M && global_col < params.N)
    {
        size_t output_index = batch_offset_c + global_row * params.strideC[params.ndC - 2]
                                           + global_col * params.strideC[params.ndC - 1];
        params.C[output_index] = sum;
    }
}

extern "C" int cuda_batched_matmul_nd_launcher(
    float *a, float *b, float *c,
    int *shape_a, size_t *stride_a, int nd_a,
    int *shape_b, size_t *stride_b, int nd_b,
    int *shape_c, size_t *stride_c, int nd_c)
{
    if (nd_a < 2 || nd_b < 2 || nd_c < 2 ||
        nd_a > MAX_DIMS || nd_b > MAX_DIMS || nd_c > MAX_DIMS)
        return 0;

    MatMulParamsND params = {};
    if (cudaMemcpy(params.shapeA, shape_a, nd_a * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(params.strideA, stride_a, nd_a * sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(params.shapeB, shape_b, nd_b * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(params.strideB, stride_b, nd_b * sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(params.shapeC, shape_c, nd_c * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(params.strideC, stride_c, nd_c * sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0;

    if (params.shapeA[nd_a - 1] != params.shapeB[nd_b - 2])
        return 0;

    int rows = params.shapeA[nd_a - 2];
    int cols = params.shapeB[nd_b - 1];
    int inner = params.shapeA[nd_a - 1];
    if (rows <= 0 || cols <= 0 || inner <= 0 ||
        params.shapeC[nd_c - 2] != rows || params.shapeC[nd_c - 1] != cols)
        return 0;

    int batches = 1;
    for (int i = 0; i < nd_c - 2; i++)
    {
        if (params.shapeC[i] <= 0 || batches > 65535 / params.shapeC[i])
            return 0;
        batches *= params.shapeC[i];
    }

    params.A = a;
    params.B = b;
    params.C = c;
    params.ndA = nd_a;
    params.ndB = nd_b;
    params.ndC = nd_c;
    params.M = rows;
    params.N = cols;
    params.K = inner;
    params.total_batches = batches;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid = cuda_grid_2d(cols, rows, TILE_SIZE, TILE_SIZE, batches);
    matmul_nd_tiled_kernel<<<grid, block>>>(params);
    return cuda_launch_status() == cudaSuccess;
}

extern "C" int cuda_matmul_launcher(float *a, float *b, float *c,
                                     int m, int n, int k,
                                     size_t a_stride0, size_t a_stride1,
                                     size_t b_stride0, size_t b_stride1,
                                     size_t c_stride0, size_t c_stride1)
{
    if (m <= 0 || n <= 0 || k <= 0)
        return 0;

    dim3 block(32, 32);
    dim3 grid = cuda_grid_2d(k, m, 32, 32);
    matmul_kernel<<<grid, block>>>(a, b, c, m, n, k,
                                   a_stride0, a_stride1, b_stride0, b_stride1,
                                   c_stride0, c_stride1);
    return cuda_launch_status() == cudaSuccess;
}