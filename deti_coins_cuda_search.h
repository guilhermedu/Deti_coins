#if USE_CUDA > 0
#ifndef DETI_COINS_CUDA_SEARCH
#define DETI_COINS_CUDA_SEARCH

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "md5.h"

static void deti_coins_cuda_search(u32_t n_random_words)
{
    u32_t idx, custom_word_1, custom_word_2;
    u64_t n_attempts, n_coins;
    void *cu_params[6];
    CUdeviceptr device_data, device_hash, device_coins_storage;
    u32_t *host_data;
    CUfunction cu_kernel;
    CUmodule cu_module;
    CUdevice cu_device;
    CUcontext cu_context;
    CUdeviceptr device_coin;

    // Inicializa palavras customizadas
    custom_word_1 = custom_word_2 = 0x20202020u;

    // Inicializa o Driver API e cria o contexto
    CU_CALL(cuInit, (0)); // Inicializa a API CUDA
    CU_CALL(cuDeviceGet, (&cu_device, 0)); // Obtém o dispositivo CUDA
    CU_CALL(cuCtxCreate, (&cu_context, 0, cu_device)); // Cria o contexto CUDA

    // Carrega o módulo CUDA e obtém o kernel
    CU_CALL(cuModuleLoad, (&cu_module, "deti_coins_cuda_kernel_search.cubin"));
    CU_CALL(cuModuleGetFunction, (&cu_kernel, cu_module, "cuda_md5_kernel_search"));

    // Aloca memória no dispositivo
    CU_CALL(cuMemAlloc, (&device_data, 1024 * sizeof(u32_t)));
    CU_CALL(cuMemAlloc, (&device_hash, 1024 * sizeof(u32_t)));
    CU_CALL(cuMemAlloc, (&device_coins_storage, 1024 * sizeof(u32_t)));

    // Aloca memória no host
    host_data = (u32_t *)malloc(1024 * sizeof(u32_t));
    if (!host_data) {
        fprintf(stderr, "Falha na alocação de memória no host\n");
        exit(EXIT_FAILURE);
    }

    // Inicializa o loop de busca
    for (n_attempts = n_coins = 0ul; stop_request == 0; n_attempts += (64ul << 20)) {
        // Prepara os dados no host
        host_data[0] = 1u; // A posição 0 armazena o índice da próxima posição livre
        CU_CALL(cuMemcpyHtoD, (device_data, host_data, 1024 * sizeof(u32_t)));

        // Configura os parâmetros do kernel
        cu_params[0] = (void *)&device_data;
        cu_params[1] = (void *)&device_hash;
        cu_params[2] = (void *)&device_coin;             
        cu_params[3] = (void *)&device_coins_storage;
        cu_params[4] = (void *)&custom_word_1;
        cu_params[5] = (void *)&custom_word_2;


        // Lança o kernel
        CU_CALL(cuLaunchKernel, (cu_kernel,
                         10, 1, 1,   // Grid dimensions
                         64, 1, 1,   // Block dimensions
                         0, 0,       // Shared memory e stream
                         cu_params, 0));
        // Copia os resultados de volta para o host
        CU_CALL(cuMemcpyDtoH, (host_data, device_data, 1024 * sizeof(u32_t)));

        // Processa os resultados no host
        for (idx = 1u; idx < host_data[0]; idx += 13u) {
            n_coins++;
            printf("Found DETI coin: ");
            for (int i = 0; i < 13; i++) {
                printf("%08x ", host_data[idx + i]);
            }
            printf("\n");
        }
    }

    // Libera memória do dispositivo
    CU_CALL(cuMemFree, (device_data));
    CU_CALL(cuMemFree, (device_hash));
    CU_CALL(cuMemFree, (device_coins_storage));

    // Libera memória do host
    free(host_data);

    // Destroi o contexto CUDA
    CU_CALL(cuCtxDestroy, (cu_context));
}

#endif // DETI_COINS_CUDA_SEARCH
#endif // USE_CUDA > 0
