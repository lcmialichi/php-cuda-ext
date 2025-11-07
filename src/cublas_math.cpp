#include <cuda_runtime.h>
#include <cublas_math.h>
#include <cublas_v2.h>
#include <cstring>

// Handle global do cuBLAS
static cublasHandle_t cublas_handle = NULL;

extern "C"
{
    int cuda_wrapper_init_blas()
    {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        return (status == CUBLAS_STATUS_SUCCESS) ? 1 : 0;
    }

    void cuda_wrapper_shutdown_blas()
    {
        if (cublas_handle != NULL)
        {
            cublasDestroy(cublas_handle);
            cublas_handle = NULL;
        }
    }

    int cuda_wrapper_matrix_create(int rows, int cols, float **matrix)
    {
        cudaError_t error = cudaMalloc(matrix, rows * cols * sizeof(float));
        return (error == cudaSuccess) ? 1 : 0;
    }

    void cuda_wrapper_matrix_free(float *matrix)
    {
        if (matrix != NULL)
        {
            cudaFree(matrix);
        }
    }

    int cuda_wrapper_matrix_multiply(const float *A, const float *B, float *C,
                                     int m, int n, int k)
    {
        if (cublas_handle == NULL)
            return 0;

        float alpha = 1.0f;
        float beta = 0.0f;

        cublasStatus_t status = cublasSgemm(cublas_handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            n, m, k, 
                                            &alpha,
                                            B, n,
                                            A, k,
                                            &beta,
                                            C, n);

        return (status == CUBLAS_STATUS_SUCCESS) ? 1 : 0;
    }

    int cuda_wrapper_matrix_add(const float *A, const float *B, float *C,
                                int rows, int cols)
    {
        int size = rows * cols;

        cudaError_t error = cudaMemcpy(C, A, size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            return 0;

        float alpha = 1.0f;
        cublasStatus_t status = cublasSaxpy(cublas_handle, size, &alpha, B, 1, C, 1);

        return (status == CUBLAS_STATUS_SUCCESS) ? 1 : 0;
    }

    int cuda_wrapper_matrix_divide(const float *A, const float *B, float *C,
                                   int rows, int cols)
    {
        int size = rows * cols;

        float *h_A = (float *)malloc(size * sizeof(float));
        float *h_B = (float *)malloc(size * sizeof(float));

        if (!h_A || !h_B)
            return 0;

        cudaMemcpy(h_A, A, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B, B, size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < size; i++)
        {
            h_A[i] = (h_B[i] != 0.0f) ? h_A[i] / h_B[i] : 0.0f;
        }

        cudaError_t error = cudaMemcpy(C, h_A, size * sizeof(float), cudaMemcpyHostToDevice);

        free(h_A);
        free(h_B);

        return (error == cudaSuccess) ? 1 : 0;
    }

}