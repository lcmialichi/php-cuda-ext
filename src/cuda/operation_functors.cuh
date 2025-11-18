#ifndef OPERATION_FUNCTORS_CUH
#define OPERATION_FUNCTORS_CUH

#include <cuda_runtime.h>
#include <math.h>

struct AddOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return a + b;
    }
};

struct SubOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return a - b;
    }
};

struct MulOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return a * b;
    }
};

struct DivOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return (fabsf(b) > 1e-12f) ? a / b : 0.0f;
    }
};

struct PowOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return powf(a, b);
    }
};

struct GreaterOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return (a > b) ? 1.0f : 0.0f;
    }
};

struct LessOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return (a < b) ? 1.0f : 0.0f;
    }
};

struct EqualOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        const float epsilon = 1e-6f;
        return (fabsf(a - b) < epsilon) ? 1.0f : 0.0f;
    }
};

struct NotEqualOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        const float epsilon = 1e-6f;
        return (fabsf(a - b) >= epsilon) ? 1.0f : 0.0f;
    }
};

struct GreaterEqualOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return (a >= b) ? 1.0f : 0.0f;
    }
};

struct LessEqualOp
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return (a <= b) ? 1.0f : 0.0f;
    }
};

struct ExpOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return expf(a);
    }
};

struct SqrtOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return sqrtf(a);
    }
};

struct LogOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return logf(a);
    }
};

struct SinOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return sinf(a);
    }
};

struct CosOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return cosf(a);
    }
};

struct TanOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return tanf(a);
    }
};

struct AbsOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return fabsf(a);
    }
};

struct NegOp
{
    __device__ __forceinline__ float operator()(float a) const
    {
        return -a;
    }
};

#endif