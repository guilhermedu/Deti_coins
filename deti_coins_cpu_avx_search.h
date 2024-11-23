#ifndef DETI_COINS_CPU_AVX_SEARCH
#define DETI_COINS_CPU_AVX_SEARCH

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>  // Para rand(), srand()
#include <time.h>    // Para time()
#include <immintrin.h>
#include "md5_cpu_avx.h"
#include "cpu_utilities.h"

typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef uint8_t u08_t;

extern volatile int stop_request;


static inline u32_t random_printable_u32() {
    return ((rand() % 95 + 0x20) << 0) |
           ((rand() % 95 + 0x20) << 8) |
           ((rand() % 95 + 0x20) << 16) |
           ((rand() % 95 + 0x20) << 24);
}

static void deti_coins_cpu_avx_search(uint32_t n_random_words) {
    u64_t n_attempts = 0, n_coins = 0;
    u32_t data[13][4] __attribute__((aligned(32)));
    u32_t hash[4][4];
    u32_t v1[4], v2[4];
    u32_t coin[13];

    srand(time(NULL));  // Semente para o gerador aleatório


    for (int j = 0; j < 4; j++) {
        data[0][j] = 0x49544544;  // "DETI" em little-endian
        data[1][j] = 0x696F6320;  // "coi " em little-endian
        data[2][j] = 0x6E20206E;  // "n  n" em little-endian
        data[12][j] = 0x0A202020; // '\n' seguido de espaços
    }

    // Loop principal
    while (!stop_request) {
        for (int i = 0; i < 4; i++) {
            v1[i] = random_printable_u32();
            v2[i] = random_printable_u32();

            // Atualiza os campos aleatórios em `data`
            data[3][i] = v1[i];
            data[4][i] = v2[i];
        }

        // Calcula os hashes MD5 para as 4 lanes
        md5_cpu_avx((v4si *)data, (v4si *)hash);

        // Verifica se algum hash é válido
        for (int lane = 0; lane < 4; lane++) {
            if (hash[3][lane] == 0) {
                // Salva o coin encontrado
                for (int i = 0; i < 13; i++) {
                    coin[i] = data[i][lane];
                }
                save_deti_coin(coin);
                n_coins++;
            }
        }

        n_attempts += 4;  
    }

    // Salva e exibe o total de coins encontrados
    STORE_DETI_COINS();
    printf("deti_coins_cpu_avx_search: %lu DETI coin%s found in %lu attempts\n",
           n_coins, n_coins == 1 ? "" : "s", n_attempts);
}

#endif
