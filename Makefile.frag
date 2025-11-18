CUDA_SRCS = src/cuda/cuda_kernels.cu src/cuda/broadcast_ops.cu src/cuda/scalar_ops.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

NVCC_FLAGS = -arch=sm_60 -O2 -Xcompiler -fPIC

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

libcudakernels.a: $(CUDA_OBJS)
	ar rcs $@ $^