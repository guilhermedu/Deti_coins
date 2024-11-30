#if USE_CUDA > 0
#ifndef DETI_COINS_CUDA_SEARCH
#define DETI_COINS_CUDA_SEARCH

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "search_utilities.h"
#include <stdlib.h> // Necessário para a função rand()

// Gera um valor aleatório de 32 bits com caracteres ASCII imprimíveis
static u32_t random_value_to_try_ascii() {
    u32_t value = 0;
    for (int i = 0; i < 4; i++) {
        u08_t random_char = (u08_t)(0x20 + (rand() % (0x7F - 0x20)));
        value |= ((u32_t)random_char) << (i * 8);
    }
    return value;
}

static void deti_coins_cuda_search(u32_t n_random_words)
{
    u32_t idx, max_idx, random_word, custom_word_1, custom_word_2;
    u64_t n_attempts, n_coins;
    void *cu_params[4];
    
    random_word = (n_random_words == 0u) ? 0x20202020u : random_value_to_try_ascii();
    // Inicializa palavras customizadas
    custom_word_1 = custom_word_2 = 0x20202020u;

    initialize_cuda(0, "deti_coins_cuda_kernel_search.cubin", "deti_coins_cuda_kernel_search", 1024u, 0);

    max_idx = 1u;
    // Inicializa o loop de busca
    for (n_attempts = n_coins = 0ul; stop_request == 0; n_attempts += (64ul << 20)) {
        // Prepara os dados no host
        host_data[0] = 1u; // A posição 0 armazena o índice da próxima posição livre
        CU_CALL(cuMemcpyHtoD, (device_data, (void *)host_data, 1024 * sizeof(u32_t)));

        
        // Configura os parâmetros do kernel
        cu_params[0] = (void *)&device_data;
        cu_params[1] = (void *)&random_word;
        cu_params[2] = (void *)&custom_word_1;             
        cu_params[3] = (void *)&custom_word_2;

        // Lança o kernel
        CU_CALL(cuLaunchKernel, (cu_kernel,
                         (1u << 20)/128u, 1u, 1u,   // Grid dimensions
                         128u, 1u, 1u,   // Block dimensions
                         0u, (CUstream)0,       // Shared memory e stream
                         &cu_params[0], NULL));
        // Copia os resultados de volta para o host
        CU_CALL(cuMemcpyDtoH, ((void *)host_data, device_data, 1024 * sizeof(u32_t)));
        //printf("random_word = %08X, custom_word_1 = %08X, custom_word_2 = %08X\n", random_word, custom_word_1, custom_word_2);
        if (host_data[0] > max_idx)
            max_idx = host_data[0];
        for (idx = 1u; idx < host_data[0] && idx <= 1024 - 13u; idx += 13u) {
            if (idx <= 1024 - 13u) {
                //printf("host_data[%u] = %08X\n", idx, host_data[idx]);
                save_deti_coin(&host_data[idx]);
                n_coins++;
            } else {
                fprintf(stderr, "deti_coins_cuda_search: wasted DETI coin \n");
            }
        }
        if (custom_word_1 != 0x7E7E7E7Eu) {
            //printf("random_word = %08X, custom_word_1 = %08X, custom_word_2 = %08X\n", random_word, custom_word_1, custom_word_2);
            next_value_to_try(custom_word_1); 
        } else {
            custom_word_1 = 0x20202020u;
            next_value_to_try(custom_word_2);
        }
        
    }   
    STORE_DETI_COINS();
    printf("deti_coins_cpu_search: %lu DETI coin%s found in %lu attempt%s (expected %.2f coins)\n",
        n_coins, (n_coins == 1ul) ? "" : "s",
        n_attempts, (n_attempts == 1ul) ? "" : "s",
        (double)n_attempts / (double)(1ul << 32));
}

#endif // DETI_COINS_CUDA_SEARCH
#endif // USE_CUDA > 0