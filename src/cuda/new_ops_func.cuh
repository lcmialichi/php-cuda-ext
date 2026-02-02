#ifndef NEW_OPS_FUNC_CUH
#define NEW_OPS_FUNC_CUH

#include <cuda_runtime.h>
#include <cmath>

template <typename T>
struct AddOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return a + b; }
};

template <typename T>
struct SubOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return a - b; }
};

template <typename T>
struct MulOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return a * b; }
};

template <typename T>
struct DivOpT
{
    static __device__ __forceinline__ T apply(T a, T b)
    {
        if constexpr (std::is_integral<T>::value)
            return b == 0 ? 0 : a / b;
        else
            return a / b;
    }
};

template <typename T>
struct PowOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return powf((float)a, (float)b); }
};

template <typename T>
struct GTOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a > b) ? (T)1 : (T)0; }
};

template <typename T>
struct LTOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a < b) ? (T)1 : (T)0; }
};

template <typename T>
struct EQOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a == b) ? (T)1 : (T)0; }
};

template <typename T>
struct NEOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a != b) ? (T)1 : (T)0; }
};

template <typename T>
struct GEOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a >= b) ? (T)1 : (T)0; }
};

template <typename T>
struct LEOpT
{
    static __device__ __forceinline__ T apply(T a, T b) { return (a <= b) ? (T)1 : (T)0; }
};

#endif