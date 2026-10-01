CUDA_SRCS = src/cuda/float_kernels.cu src/cuda/activation_kernels.cu src/cuda/matmul_kernels.cu src/cuda/concat_kernels.cu src/cuda/where_kernels.cu src/cuda/broadcast_ops.cu src/cuda/scalar_ops.cu src/cuda/unary_ops.cu src/cuda/reduction_ops.cu src/cuda/factory_kernels.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

NVCC_FLAGS = -arch=$(CUDA_ARCH_FLAG) -O3 --use_fast_math -Xcompiler -fPIC

./cuda.la: libcudakernels.a

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

libcudakernels.a: $(CUDA_OBJS)
	rm -f $@
	ar rcs $@ $^
