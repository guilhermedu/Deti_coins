#
# Makefile for the DETI Coins project
# TODOS OS MAKEFILES JUNTOS

#
# CUDA installation directory --- /usr/local/cuda or $(CUDA_HOME)
#
CUDA_DIR = /usr/local/cuda

OPENCL_DIR = $(CUDA_DIR)

#
# CUDA device architecture
#
CUDA_ARCH = sm_61

#
# Compiler flags
#
BASE_CFLAGS = -Wall -O2 -mavx2
OPENMP_CFLAGS = $(BASE_CFLAGS) -fopenmp
CUDAFLAGS = -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall
LDFLAGS = -fopenmp

#
# Enable SIMD + OpenMP functionality
#
SIMD_OPENMP_FLAGS = -DDETI_COINS_CPU_AVX_SIMD_OPENMP_SEARCH -DDETI_COINS_CPU_AVX2_SIMD_OPENMP_SEARCH

OPENCL_KERNELS = deti_coins_opencl_kernel.cl deti_coins_opencl_search.cl


#
# Source code files
#
SRC       = deti_coins.c
H_FILES   = cpu_utilities.h
H_FILES  += md5.h md5_test_data.h md5_cpu.h md5_cpu_avx.h md5_cpu_avx2.h md5_cpu_neon.h
H_FILES  += deti_coins_vault.h deti_coins_cpu_search.h deti_coins_cpu_avx_search.h deti_coins_cpu_avx2_search.h deti_coins_cpu_avx2_SIMD_OPENMP.h deti_coins_cpu_avx_SIMD_OPENMP.h deti_coins_opencl_search.h
H_FILES  += cuda_driver_api_utilities.h md5_cuda.h

#
# Clean up
#
clean:
	rm -f a.out
	rm -f deti_coins_intel
	rm -f deti_coins_intel_openmp
	rm -f deti_coins_apple
	rm -f deti_coins_intel_cuda md5_cuda_kernel.cubin deti_coins_cuda_kernel_search.cubin
	rm -f deti_coins_server deti_coins_client
	rm -f deti_coins_server_avx2 deti_coins_client_avx2
	rm -f deti_coins.html deti_coins.js deti_coins.wasm
	rm -f deti_coins_opencl_search
	rm -f deti_coins_opencl_kernel.bin deti_coins_opencl_search.bin
	rm -f deti_coins_intel_avx_simd
	rm -f deti_coins_intel_avx2_simd


#
# Compile for Intel/AMD processors without OpenMP or CUDA
#
deti_coins_intel: $(SRC) $(H_FILES)
	cc $(BASE_CFLAGS) -DUSE_CUDA=0 $(SRC) -o deti_coins_intel

deti_coins_intel_avx_simd: $(SRC) $(H_FILES)
	cc $(BASE_CFLAGS) -DDETI_COINS_CPU_AVX_SIMD -DUSE_CUDA=0 $(SRC) -o deti_coins_intel_avx_simd

deti_coins_intel_avx2_simd: $(SRC) $(H_FILES)
	cc $(BASE_CFLAGS) -DDETI_COINS_CPU_AVX2_SIMD -DUSE_CUDA=0 $(SRC) -o deti_coins_intel_avx2_simd

#
# Compile for Intel/AMD processors with OpenMP support
#
deti_coins_intel_openmp: $(SRC) $(H_FILES)
	cc $(OPENMP_CFLAGS) $(SIMD_OPENMP_FLAGS) -DUSE_CUDA=0 $(SRC) -o deti_coins_intel_openmp $(LDFLAGS)

#
# Compilation for Apple silicon without CUDA
#
deti_coins_apple: $(SRC) $(H_FILES)
	cc $(BASE_CFLAGS) -DUSE_CUDA=0 $(SRC) -o deti_coins_apple

#
# Compile for Intel/AMD processors with CUDA
#
deti_coins_intel_cuda: $(SRC) $(H_FILES) $(C_FILES) md5_cuda_kernel.cubin deti_coins_cuda_kernel_search.cubin
	cc -Wall -O2 -mavx2 -DUSE_CUDA=1 -I$(CUDA_DIR)/include $(SRC) -o deti_coins_intel_cuda -L$(CUDA_DIR)/lib64 -lcuda

md5_cuda_kernel.cubin: md5.h md5_cuda_kernel.cu
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin md5_cuda_kernel.cu -o md5_cuda_kernel.cubin

deti_coins_cuda_kernel_search.cubin: md5.h deti_coins_cuda_kernel_search.cu
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin deti_coins_cuda_kernel_search.cu -o deti_coins_cuda_kernel_search.cubin


deti_coins_server: $(SRC) $(H_FILES)
	cc $(OPENMP_CFLAGS) -DDETI_COINS_SERVER $(SRC) -o deti_coins_server $(LDFLAGS)

deti_coins_client: deti_coins.c $(H_FILES)
	cc $(OPENMP_CFLAGS) -DDETI_COINS_CLIENT $(SRC) -o deti_coins_client $(LDFLAGS)

deti_coins_server_avx2: $(SRC) $(H_FILES)
	cc $(OPENMP_CFLAGS) -DDETI_COINS_SERVER_AVX2 $(SRC) -o deti_coins_server_avx2 $(LDFLAGS)

deti_coins_client_avx2: deti_coins.c $(H_FILES)
	cc $(OPENMP_CFLAGS) -DDETI_COINS_CLIENT_AVX2 $(SRC) -o deti_coins_client_avx2 $(LDFLAGS)

#
# Compile deti_coins.c into WebAssembly
#
webAssembly: $(SRC) $(H_FILES)
	emcc -Wall -O2 $(SRC) -o deti_coins.html --emrun

#
# Compile OpenCL Kernels
#
deti_coins_opencl_kernel.bin: md5.h deti_coins_opencl_kernel.cl
	clang -x cl -cl-std=CL2.0 -I /usr/include -o deti_coins_opencl_kernel.bin -c deti_coins_opencl_kernel.cl
#
# Compile deti_coins.c with OpenCL support
#
deti_coins_opencl_search: $(SRC) $(H_FILES) deti_coins_opencl_kernel.bin
	cc -Wall -O2 -DUSE_OPENCL=1 -DDETI_COINS_OPENCL_SEARCH $(SRC) -o deti_coins_opencl_search -lOpenCL


webAssembly: $(SRC) $(H_FILES)
	emcc -Wall -O2 $(SRC) -o deti_coins.c --emrun