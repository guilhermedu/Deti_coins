#include "md5_cuda.h"

typedef unsigned int u32_t;



// Kernel CUDA para busca de moedas DETI
extern "C" __global__ __launch_bounds__(128, 1)void deti_coins_cuda_kernel_search(u32_t *interleaved_data, u32_t interleaved_hash, u32_t *found_coins_buffer, u32_t buffer_size, u32_t custom_word_1, u32_t custom_word_2,u32_t coin[13]) {
    u32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x; // Índice global único da thread
 



    md5_cuda(interleaved_data,interleaved_hash);

    // Validar o hash (condição para uma moeda DETI válida)
    if (hash[3] == 0) { // Exemplo de condição para uma moeda válida
        int idx = atomicAdd(&found_coins_buffer[0], 1); // Reservar espaço no buffer
        if (idx < buffer_size - 1) {
            for (int i = 0; i < 13; i++) {
                found_coins_buffer[1 + idx * 13 + i] = coin_template[i];
            }
        }
    }
}
