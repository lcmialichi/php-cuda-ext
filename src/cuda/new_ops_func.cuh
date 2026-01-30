#include <cuda_runtime.h>
#include <cmath>

template <typename T>
struct AddOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return a + b; }
};

template <typename T>
struct SubOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return a - b; }
};

template <typename T>
struct MulOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return a * b; }
};

template <typename T>
struct DivOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const
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
    __device__ __forceinline__ T operator()(const T a, const T b) const { return powf((float)a, (float)b); }
};

template <typename T>
struct GTOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return (a > b) ? (T)1 : (T)0; }
};

template <typename T>
struct LTOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return (a < b) ? (T)1 : (T)0; }
};

template <typename T>
struct EQOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return (a == b) ? (T)1 : (T)0; }
};

template <typename T>
struct NEOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return (a != b) ? (T)1 : (T)0; }
};

template <typename T>
struct GEOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return (a >= b) ? (T)1 : (T)0; }
};

template <typename T>
struct LEOpT
{
    __device__ __forceinline__ T operator()(const T a, const T b) const { return (a <= b) ? (T)1 : (T)0; }
};