# Deti_coins

## AVX
```
make deti_coins_intel
```
```
./deti_coins_intel -s1 180
```

## AVX2
```
make deti_coins_intel
```
```
./deti_coins_intel -s2 180
```

## CUDA
```
make_deti_coins_intel_cuda
```
```
./deti_coins_intel_cuda -s4 180
```

## AVX_SIMD
```
make deti_coins_intel_avx_simd
```
```
./deti_coins_intel_avx_simd -se 180
```

## AVX2_SIMD
```
make deti_coins_intel_avx2_simd
```
```
./deti_coins__intel_avx2_simd -sf 180
```

## AVX OPENMP
```
make deti_coins_intel_openmp
```
```
./deti_coins_intel_openmp -s5 180
```

## AVX2 OPENMP
```
make deti_coins_intel_openmp
```
```
./deti_coins_intel_openmp -s6 180
```

## AVX Server
```
make deti_coins_server
```
```
./deti_coins_server -s7 180
```

## AVX Client
```
make deti_coins_client
```
```
./deti_coins_client -s8 180 127.0.0.1:5000
```

## AVX2 Server
```
make deti_coins_server_avx2
```
```
./deti_coins_server -sa 180
```

## AVX2 Client
```
make deti_coins_client
```
```
./deti_coins_client -sb 180 127.0.0.1:5000
```

## Web Assembly
```
make webAssembly
```
```
emrun deti_coins.html -- -s0 180
```

## OPENCL
```
make deti_coins_opencl_search
```
```
./deti_coins_opencl_search -sd 180
```