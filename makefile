#
# Makefile for the deti_coins project
#

#
# CUDA installation directory --- /usr/local/cuda or $(CUDA_HOME)
#
CUDA_DIR = /usr/local/cuda

#
# OpenCL installation directory (for an NVidia graphics card, same as CUDA)
#


#
# CUDA device architecture
#   GeForce GTX 1660 Ti --- sm_75
#   RTX A2000 Ada --------- sm_86
#   RTX A6000 Ada --------- sm_86
#   RTX 4070 -------------- sm_89
#
CUDA_ARCH = sm_61

#
# Source code files
#
SRC       = deti_coins.c
H_FILES   = cpu_utilities.h
H_FILES  += md5.h md5_test_data.h md5_cpu.h md5_cpu_avx.h md5_cpu_neon.h
H_FILES  += deti_coins_vault.h deti_coins_cpu_search.h deti_coins_opencl_search.h
C_FILES   = cuda_driver_api_utilities.h md5_cuda.h

#
# OpenCL Kernel Files
#
OPENCL_KERNELS = deti_coins_opencl_kernel.cl deti_coins_opencl_search.cl

#
# Clean up
#
clean:
	rm -f a.out
	rm -f deti_coins_intel
	rm -f deti_coins_apple
	rm -f deti_coins_intel_cuda md5_cuda_kernel.cubin deti_coins_cuda_kernel_search.cubin
	rm -f deti_coins_opencl_kernel.bin deti_coins_opencl_search.bin
	rm -f deti_coins_opencl_search
	rm -f deti_coins.html deti_coins.js deti_coins.wasm

#
# Compile for Intel/AMD processors without CUDA
#
deti_coins_intel: $(SRC) $(H_FILES)
	cc -Wall -O2 -mavx2 -DUSE_CUDA=0 $(SRC) -o deti_coins_intel

#
# Compilation for Apple Silicon without CUDA
#
deti_coins_apple: $(SRC) $(H_FILES)
	cc -Wall -O2 -DUSE_CUDA=0 $(SRC) -o deti_coins_apple

#
# Compile for Intel/AMD processors with CUDA
#
deti_coins_intel_cuda: $(SRC) $(H_FILES) $(C_FILES) md5_cuda_kernel.cubin deti_coins_cuda_kernel_search.cubin
	cc -Wall -O2 -mavx2 -DUSE_CUDA=1 -I$(CUDA_DIR)/include $(SRC) -o deti_coins_intel_cuda -L$(CUDA_DIR)/lib64 -lcuda

#
# Generate CUDA kernels
#
md5_cuda_kernel.cubin: md5.h md5_cuda_kernel.cu
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin md5_cuda_kernel.cu -o md5_cuda_kernel.cubin

deti_coins_cuda_kernel_search.cubin: md5.h deti_coins_cuda_kernel_search.cu
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin deti_coins_cuda_kernel_search.cu -o deti_coins_cuda_kernel_search.cubin

#
# Compile OpenCL Kernels
#
deti_coins_opencl_kernel.bin: md5.h deti_coins_opencl_kernel.cl
	clang -x cl -cl-std=CL2.0 -I/include -o deti_coins_opencl_kernel.bin -c deti_coins_opencl_kernel.cl



#
# Compile deti_coins.c with OpenCL support
#
deti_coins_opencl_search: $(SRC) $(H_FILES) deti_coins_opencl_kernel.bin 
	cc -Wall -O2 -DUSE_OPENCL=1 -DDETI_COINS_OPENCL_SEARCH=1 $(SRC) -o deti_coins_opencl_search -L/lib -lOpenCL


#
# Compile deti_coins.c into WebAssembly
#
webAssembly: $(SRC) $(H_FILES)
	emcc -Wall -O2 $(SRC) -o deti_coins.html
