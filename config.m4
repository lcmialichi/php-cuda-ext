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
    PHP_ADD_LIBRARY(stdc++, 1, CUDA_SHARED_LIBADD)
    
    PHP_ADD_LIBRARY_WITH_PATH(cudnn, /usr/local/cuda/lib64, CUDA_SHARED_LIBADD)
    PHP_ADD_LIBRARY_WITH_PATH(cublas, /usr/local/cuda/lib64, CUDA_SHARED_LIBADD)
    PHP_ADD_LIBRARY_WITH_PATH(cudart, /usr/local/cuda/lib64, CUDA_SHARED_LIBADD)

    CXXFLAGS="$CXXFLAGS -O2"
    CFLAGS="$CFLAGS -O2"
    
    dnl Lista de arquivos CUDA
    CUDA_FILES="src/cuda_kernels.cu src/broadcast_ops.cu src/scalar_ops.cu"

    AC_MSG_CHECKING([for CUDA kernels])
    for f in $CUDA_FILES; do
        $NVCC -arch=sm_60 -O2 -Xcompiler -fPIC -c $f -o ${f%.cu}.o || {
            AC_MSG_ERROR([Failed to compile $f])
        }
    done

    AC_MSG_RESULT([yes])

    ar rcs libcudakernels.a src/*.o

    PHP_EVAL_LIBLINE([-L. -lcudakernels], CUDA_SHARED_LIBADD)
    
    PHP_SUBST(CUDA_SHARED_LIBADD)
    
    PHP_NEW_EXTENSION(cuda, src/cuda.c src/cuda_wrapper.cpp src/cuda_array.c src/cuda_array_wrapper.cpp src/tensor.c, $ext_shared)
    PHP_ADD_MAKEFILE_FRAGMENT(makefile.frag, $ext_srcdir)
fi