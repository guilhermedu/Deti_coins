#ifndef DETI_COINS_CPU_AVX_SIMD_H
#define DETI_COINS_CPU_AVX_SIMD_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include "md5_cpu_avx.h"
#include "cpu_utilities.h"
#include "deti_coins_vault.h"

typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef uint8_t u08_t;

extern volatile int stop_request;


#define NUM_LANES 4 
#define NUM_VARIABLES 4

// Global state for the xorshift32 PRNG
static __m128i xorshift32_state_avx1[NUM_VARIABLES];

// Initialize the PRNG state with different seeds for each lane and variable
static void initialize_xorshift32_state_avx1() {
    for (int var = 0; var < NUM_VARIABLES; var++) {
        xorshift32_state_avx1[var] = _mm_set_epi32(
            (uint32_t)time(NULL) + var * NUM_LANES + 3,
            (uint32_t)time(NULL) + var * NUM_LANES + 2,
            (uint32_t)time(NULL) + var * NUM_LANES + 1,
            (uint32_t)time(NULL) + var * NUM_LANES
        );
    }
}

// Vectorized xorshift32 PRNG using AVX1
static inline __m128i xorshift32_avx1(__m128i *state) {
    __m128i x = *state;

    x = _mm_xor_si128(x, _mm_slli_epi32(x, 13));
    x = _mm_xor_si128(x, _mm_srli_epi32(x, 17));
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 5));

    *state = x;
    return x;
}

// Function to generate four u32_t values with printable ASCII characters using AVX1
static void random_printable_u32_avx1_simd(u32_t *v, __m128i *state) {
    __m128i rand_nums = xorshift32_avx1(state);

    // Extract the bytes from the 128-bit vector
    __m128i bytes_16 = _mm_and_si128(rand_nums, _mm_set1_epi32(0xFF)); // Keep the lower 8 bits only

    // Multiply by 95 to map to the range [0, 94]
    __m128i prod = _mm_mullo_epi16(bytes_16, _mm_set1_epi16(95));

    // Get high byte to scale down to [0, 94]
    __m128i high_bytes = _mm_srli_epi16(prod, 8);

    // Add 0x20 to shift to printable ASCII range [0x20, 0x7E]
    __m128i ascii_chars = _mm_add_epi16(high_bytes, _mm_set1_epi16(0x20));

    // Pack 4 uint16_t back into 4 uint8_t
    __m128i ascii_chars_packed = _mm_packus_epi16(ascii_chars, _mm_setzero_si128());

    // Store the result into the array 'v'
    _mm_storeu_si128((__m128i *)v, ascii_chars_packed);
}


static void deti_coins_cpu_avx_search_simd(uint32_t n_random_words) {
    u64_t n_attempts = 0, n_coins = 0;
    u32_t data[13][NUM_LANES] __attribute__((aligned(32)));
    u32_t hash[4][NUM_LANES];
    u32_t v1[NUM_LANES], v2[NUM_LANES], v3[NUM_LANES], v4[NUM_LANES];

    // Initialize the xorshift32 PRNG state
    initialize_xorshift32_state_avx1();

    // Initialization of 'data' with static values
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < NUM_LANES; j++) {
            data[i][j] = 0x20202020; // Little-endian '    '
        }
    }

    // Set the specific initial values
    for (int j = 0; j < NUM_LANES; j++) {
        data[0][j] = 0x49544544; // "DETI" in little-endian
        data[1][j] = 0x696F6320; // "coi " in little-endian
        data[2][j] = 0x6E20206E; // "n  n" in little-endian
        data[12][j] = 0x0A202020; // Newline character '\n' followed by spaces
    }

    u32_t coin[13];

    // Main loop to find DETI coins
    for (n_attempts = n_coins = 0ul; stop_request == 0; n_attempts += NUM_LANES) {
        // Generate random values for v1, v2, v3, v4 using their respective PRNG states
        random_printable_u32_avx1_simd(v1, &xorshift32_state_avx1[0]);
        random_printable_u32_avx1_simd(v2, &xorshift32_state_avx1[1]);
        random_printable_u32_avx1_simd(v3, &xorshift32_state_avx1[2]);
        random_printable_u32_avx1_simd(v4, &xorshift32_state_avx1[3]);

        // Update data arrays for each lane
        for (int i = 0; i < NUM_LANES; i++) {
            data[3][i] = v1[i];
            data[4][i] = v2[i];
            data[5][i] = v3[i];
            data[6][i] = v4[i];
        }

        // Compute MD5 hashes for all lanes
        md5_cpu_avx((v4si *)data, (v4si *)hash);

        // Process each lane separately
        for (int lane = 0; lane < NUM_LANES; lane++) {
            u32_t hash_lane[4];

            for (int i = 0; i < 4; i++) {
                hash_lane[i] = hash[i][lane];
            }

            hash_byte_reverse(hash_lane);

            u32_t n = deti_coin_power(hash_lane);

            if (n >= 32u) {
                for (int i = 0; i < 13; i++) {
                    coin[i] = data[i][lane];
                }
                save_deti_coin(coin);
                n_coins++;
            }
        }
    }

    STORE_DETI_COINS();
    printf("deti_coins_cpu_search: %lu DETI coin%s found in %lu attempt%s (expected %.2f coins)\n",
           n_coins, (n_coins == 1ul) ? "" : "s",
           n_attempts, (n_attempts == 1ul) ? "" : "s",
           (double)n_attempts / (double)(1ul << 32));
}

#endif
