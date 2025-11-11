NVCC_FLAGS = -arch=sm_60 -O2 -Xcompiler -fPIC

$(builddir)/src/cuda_kernels.o: src/cuda_kernels.cu
	@mkdir -p $(builddir)/src
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

libcudakernels.a: $(builddir)/src/cuda_kernels.o
	ar rcs $@ $<

$(builddir)/cuda.la: libcudakernels.a

clean-cuda:
	rm -f $(builddir)/src/cuda_kernels.o libcudakernels.a