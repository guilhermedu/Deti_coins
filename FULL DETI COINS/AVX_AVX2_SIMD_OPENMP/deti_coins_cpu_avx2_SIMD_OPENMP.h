#ifndef DETI_COINS_CPU_AVX2_SIMD_OPENMP_H
#define DETI_COINS_CPU_AVX2_SIMD_OPENMP_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include "md5_cpu_avx2.h"
#include "cpu_utilities.h"
#include "deti_coins_vault.h"

typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef uint8_t u08_t;

extern volatile int stop_request;

#define NUM_LANES 8
#define NUM_VARIABLES 4

static void initialize_xorshift32_state(__m256i *state) {
    // Improved seed initialization
    unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
    seed ^= (unsigned int)(uintptr_t)&state;
    seed ^= (unsigned int)clock();
    seed ^= (unsigned int)(omp_get_wtime() * 1e9);

    for (int var = 0; var < NUM_VARIABLES; var++) {
        state[var] = _mm256_set_epi32(
            seed + var * NUM_LANES + 7,
            seed + var * NUM_LANES + 6,
            seed + var * NUM_LANES + 5,
            seed + var * NUM_LANES + 4,
            seed + var * NUM_LANES + 3,
            seed + var * NUM_LANES + 2,
            seed + var * NUM_LANES + 1,
            seed + var * NUM_LANES
        );
    }
}

static inline __m256i xorshift32_avx2(__m256i *state) {
    __m256i x = *state;
    __m256i tmp;

    tmp = _mm256_slli_epi32(x, 13);
    x = _mm256_xor_si256(x, tmp);

    tmp = _mm256_srli_epi32(x, 17);
    x = _mm256_xor_si256(x, tmp);

    tmp = _mm256_slli_epi32(x, 5);
    x = _mm256_xor_si256(x, tmp);

    *state = x;
    return x;
}

static void random_printable_u32_avx2_openmp(u32_t *v, __m256i *state) {

    __m256i rand_nums = xorshift32_avx2(state);

    __m128i bytes_low = _mm256_extracti128_si256(rand_nums, 0);
    __m128i bytes_high = _mm256_extracti128_si256(rand_nums, 1);

    __m256i bytes_16_low = _mm256_cvtepu8_epi16(bytes_low);
    __m256i bytes_16_high = _mm256_cvtepu8_epi16(bytes_high);

    __m256i prod_low = _mm256_mullo_epi16(bytes_16_low, _mm256_set1_epi16(95));
    __m256i prod_high = _mm256_mullo_epi16(bytes_16_high, _mm256_set1_epi16(95));

    __m256i high_bytes_low = _mm256_srli_epi16(prod_low, 8);
    __m256i high_bytes_high = _mm256_srli_epi16(prod_high, 8);

    __m256i ascii_chars_low = _mm256_add_epi16(high_bytes_low, _mm256_set1_epi16(0x20));
    __m256i ascii_chars_high = _mm256_add_epi16(high_bytes_high, _mm256_set1_epi16(0x20));

    __m256i ascii_chars_packed = _mm256_packus_epi16(ascii_chars_low, ascii_chars_high);

    _mm256_storeu_si256((__m256i *)v, ascii_chars_packed);
}

static void deti_coins_cpu_avx2_simd_openmp_search(uint32_t n_random_words) {
    u64_t n_attempts = 0, n_coins = 0;

    #pragma omp parallel reduction(+:n_attempts, n_coins)
    {
        __m256i xorshift32_state[NUM_VARIABLES];
        u32_t data[13][NUM_LANES] __attribute__((aligned(32)));
        u32_t hash[4][NUM_LANES];
        u32_t v1[NUM_LANES], v2[NUM_LANES], v3[NUM_LANES], v4[NUM_LANES];
        u32_t coin[13];

        // Initialize PRNG state with improved seeding
        initialize_xorshift32_state(xorshift32_state);

        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < NUM_LANES; j++) {
                data[i][j] = 0x20202020;
            }
        }

        for (int j = 0; j < NUM_LANES; j++) {
            data[0][j] = 0x49544544;
            data[1][j] = 0x696F6320;
            data[2][j] = 0x6E20206E;
            data[12][j] = 0x0A202020;
        }

        while (stop_request == 0) {
            random_printable_u32_avx2_openmp(v1, &xorshift32_state[0]);
            random_printable_u32_avx2_openmp(v2, &xorshift32_state[1]);
            random_printable_u32_avx2_openmp(v3, &xorshift32_state[2]);
            random_printable_u32_avx2_openmp(v4, &xorshift32_state[3]);

            for (int i = 0; i < NUM_LANES; i++) {
                data[3][i] = v1[i];
                data[4][i] = v2[i];
                data[5][i] = v3[i];
                data[6][i] = v4[i];
            }

            md5_cpu_avx2((v8si *)data, (v8si *)hash);

            for (int lane = 0; lane < NUM_LANES; lane++) {
                if (hash[3][lane] == 0) {
                    for (int i = 0; i < 13; i++) {
                        coin[i] = data[i][lane];
                    }
                    #pragma omp critical
                    {
                        save_deti_coin(coin);
                    }
                    n_coins++;
                }
            }

            n_attempts += NUM_LANES;
        }
    }

    STORE_DETI_COINS();
    printf("deti_coins_cpu_avx2_simd_openmp_search: %lu DETI coin%s found in %lu attempt%s (expected %.2f coins)\n",
           n_coins, (n_coins == 1ul) ? "" : "s",
           n_attempts, (n_attempts == 1ul) ? "" : "s",
           (double)n_attempts / (double)(1ul << 32));
}

#endif // DETI_COINS_CPU_AVX2_SIMD_OPENMP_H
