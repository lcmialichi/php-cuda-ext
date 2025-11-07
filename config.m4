PHP_ARG_WITH(cuda, for CUDA support,
[  --with-cuda             Include CUDA support])

if test "$PHP_CUDA" != "no"; then
    AC_PATH_PROG(NVCC, nvcc, no)
    if test "$NVCC" = "no"; then
        AC_MSG_ERROR([nvcc not found - please install CUDA toolkit])
    fi

    PHP_ADD_INCLUDE(/usr/local/cuda/include)
    PHP_ADD_INCLUDE([src])
    
    PHP_NEW_EXTENSION(cuda, 
        src/cuda.c 
        src/cuda_wrapper.cpp,
        src/cublas_math.cpp,
        $ext_shared)
    
    # Links CUDA
    PHP_ADD_LIBRARY_WITH_PATH(cudart, /usr/local/cuda/lib64, CUDA_SHARED_LIBADD)
    PHP_SUBST(CUDA_SHARED_LIBADD)
fi