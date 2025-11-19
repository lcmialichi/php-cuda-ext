PHP_ARG_WITH(cuda, for CUDA support,
[  --with-cuda             Include CUDA support])

if test "$PHP_CUDA" != "no"; then
    PHP_REQUIRE_CXX()
    AC_PATH_PROG(NVCC, nvcc, no)
    if test "$NVCC" = "no"; then
        AC_MSG_ERROR([nvcc not found - please install CUDA toolkit])
    fi

    PHP_ADD_INCLUDE(/usr/local/cuda/include)
    PHP_ADD_INCLUDE([src])
    PHP_ADD_INCLUDE([src/cuda])
    PHP_ADD_INCLUDE([src/cuda_array])
    PHP_ADD_LIBRARY(stdc++, 1, CUDA_SHARED_LIBADD)
    
    PHP_ADD_LIBRARY_WITH_PATH(cudart, /usr/local/cuda/lib64, CUDA_SHARED_LIBADD)

    CXXFLAGS="$CXXFLAGS -O2"
    CFLAGS="$CFLAGS -O2"
    
    CUDA_FILES="src/cuda/cuda_kernels.cu src/cuda/broadcast_ops.cu src/cuda/scalar_ops.cu src/cuda/unary_ops.cu"

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
    src/cuda_array/tensor_fabric.c"

    PHP_NEW_EXTENSION(cuda, $SRC_FILES, $ext_shared)
    PHP_ADD_MAKEFILE_FRAGMENT(makefile.frag, $ext_srcdir)
fi