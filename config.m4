PHP_ARG_WITH(cuda, for CUDA support,
[  --with-cuda             Include CUDA support])

if test "$PHP_CUDA" != "no"; then
    PHP_REQUIRE_CXX()
    AC_PATH_PROG(NVCC, nvcc, no)
    if test "$NVCC" = "no"; then
        AC_MSG_ERROR([nvcc not found - please install CUDA toolkit])
    fi

    NVCCFLAGS="$NVCCFLAGS -use_fast_math"

    PHP_ADD_INCLUDE(/usr/local/cuda/include)
    PHP_ADD_INCLUDE([src])
    PHP_ADD_INCLUDE([src/cuda])
    PHP_ADD_INCLUDE([src/kernel])
    PHP_ADD_INCLUDE([src/cuda_array])
    PHP_ADD_LIBRARY(stdc++, 1, CUDA_SHARED_LIBADD)

    AC_DEFINE_UNQUOTED(CUDA_INCLUDE_PATH_STR, "-I$PHP_CUDA/include", [inclusion path for NVRTC JIT])
    AC_DEFINE_UNQUOTED(CUDA_CRT_INCLUDE_STR, "-I$PHP_CUDA/include/crt", [ C++ inclusion path for NVRTC JIT])

    PHP_CHECK_LIBRARY(nvrtc, nvrtcCreateProgram, [
    PHP_ADD_LIBRARY_WITH_PATH(nvrtc, $PHP_CUDA/lib64, CUDA_SHARED_LIBADD)
    ], [
        AC_MSG_ERROR([libnvrtc not found. Please set PHP_CUDA or ensure libnvrtc is installed.])
    ])

    PHP_CHECK_LIBRARY(cuda, cuInit, [
        PHP_ADD_LIBRARY_WITH_PATH(cuda, $PHP_CUDA/lib64, CUDA_SHARED_LIBADD)
    ], [
        AC_MSG_WARN([libcuda (Driver API) not explicitly found. Assuming it is available in standard path.])
        PHP_ADD_LIBRARY(cuda, CUDA_SHARED_LIBADD)
    ])

    PHP_ADD_LIBRARY_WITH_PATH(cudart, $PHP_CUDA/lib64, CUDA_SHARED_LIBADD)
    PHP_ADD_LIBRARY_WITH_PATH(curand, $PHP_CUDA/lib64, CUDA_SHARED_LIBADD)

    CXXFLAGS="$CXXFLAGS -O2"
    CFLAGS="$CFLAGS -O2"
    
    CUDA_FILES="src/cuda/cuda_kernels.cu src/cuda/broadcast_ops.cu src/cuda/scalar_ops.cu src/cuda/unary_ops.cu src/cuda/reduction_ops.cu"

    AC_MSG_CHECKING([for CUDA kernels])
    for f in $CUDA_FILES; do
        $NVCC -arch=sm_60 -O2 -Xcompiler -fPIC -c $f -o ${f%.cu}.o || {
            AC_MSG_ERROR([Failed to compile $f])
        }
    done

    AC_MSG_RESULT([yes])

    ar rcs libcudakernels.a src/cuda/*.o

    PHP_EVAL_LIBLINE([-L. -lcudakernels], CUDA_SHARED_LIBADD)
    
    PHP_SUBST(CUDA_SHARED_LIBADD)
    SRC_FILES="\
    src/cuda.c \
    src/cuda_wrapper.cpp \
    src/cuda_array/cuda_array.c \ 
    src/cuda_array/ca_private.c \
    src/tensor.c \
    src/cuda/memory_pool.c \
    src/cuda_array/tensor_fabric.c \
    src/operations.c \
    src/kernel/kernel.c \
    src/cuda_attributes.c \
    src/kernel_reflection.c \
    src/ast_cuda_compiler.c"

    PHP_NEW_EXTENSION(cuda, $SRC_FILES, $ext_shared)
    PHP_ADD_MAKEFILE_FRAGMENT(makefile.frag, $ext_srcdir)
fi