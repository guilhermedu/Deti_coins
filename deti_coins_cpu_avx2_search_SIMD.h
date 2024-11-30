#ifndef DETI_COINS_CPU_AVX2_SEARCH
#define DETI_COINS_CPU_AVX2_SEARCH

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>    
#include <immintrin.h>
#include "md5_cpu_avx2.h"
#include "cpu_utilities.h"
#include "deti_coins_vault.h"

typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef uint8_t u08_t;

extern volatile int stop_request;

#define NUM_LANES 8
#define NUM_VARIABLES 4

// Global state for the xorshift32 PRNG
static __m256i xorshift32_state[NUM_VARIABLES];

// Initialize the PRNG state with different seeds for each lane and variable
static void initialize_xorshift32_state() {
    for (int var = 0; var < NUM_VARIABLES; var++) {
        xorshift32_state[var] = _mm256_set_epi32(
            (uint32_t)time(NULL) + var * NUM_LANES + 7,
            (uint32_t)time(NULL) + var * NUM_LANES + 6,
            (uint32_t)time(NULL) + var * NUM_LANES + 5,
            (uint32_t)time(NULL) + var * NUM_LANES + 4,
            (uint32_t)time(NULL) + var * NUM_LANES + 3,
            (uint32_t)time(NULL) + var * NUM_LANES + 2,
            (uint32_t)time(NULL) + var * NUM_LANES + 1,
            (uint32_t)time(NULL) + var * NUM_LANES
        );
    }
}

// Vectorized xorshift32 PRNG using AVX2
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

// Function to generate eight u32_t values with printable ASCII characters using AVX2
static void random_printable_u32_avx2(u32_t *v, __m256i *state) {
    __m256i rand_nums = xorshift32_avx2(state);

    // Split the 256-bit vector into two 128-bit vectors
    __m128i bytes_low = _mm256_extracti128_si256(rand_nums, 0);
    __m128i bytes_high = _mm256_extracti128_si256(rand_nums, 1);

    // Convert 16 uint8_t to 16 uint16_t for low and high parts
    __m256i bytes_16_low = _mm256_cvtepu8_epi16(bytes_low);
    __m256i bytes_16_high = _mm256_cvtepu8_epi16(bytes_high);

    // Multiply by 95 to map to the range [0, 94]
    __m256i prod_low = _mm256_mullo_epi16(bytes_16_low, _mm256_set1_epi16(95));
    __m256i prod_high = _mm256_mullo_epi16(bytes_16_high, _mm256_set1_epi16(95));

    // Get high byte to scale down to [0, 94]
    __m256i high_bytes_low = _mm256_srli_epi16(prod_low, 8);
    __m256i high_bytes_high = _mm256_srli_epi16(prod_high, 8);

    // Add 0x20 to shift to printable ASCII range [0x20, 0x7E]
    __m256i ascii_chars_low = _mm256_add_epi16(high_bytes_low, _mm256_set1_epi16(0x20));
    __m256i ascii_chars_high = _mm256_add_epi16(high_bytes_high, _mm256_set1_epi16(0x20));

    // Pack 16 uint16_t back into 16 uint8_t
    __m256i ascii_chars_packed = _mm256_packus_epi16(ascii_chars_low, ascii_chars_high);

    // Store the result into the array 'v'
    _mm256_storeu_si256((__m256i *)v, ascii_chars_packed);
}

static void deti_coins_cpu_avx2_search(uint32_t n_random_words) {
    u64_t n_attempts = 0, n_coins = 0;
    u32_t data[13][NUM_LANES] __attribute__((aligned(32)));
    u32_t hash[4][NUM_LANES];
    u32_t v1[NUM_LANES], v2[NUM_LANES], v3[NUM_LANES], v4[NUM_LANES];

    // Initialize the xorshift32 PRNG state
    initialize_xorshift32_state();

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
        random_printable_u32_avx2(v1, &xorshift32_state[0]);
        random_printable_u32_avx2(v2, &xorshift32_state[1]);
        random_printable_u32_avx2(v3, &xorshift32_state[2]);
        random_printable_u32_avx2(v4, &xorshift32_state[3]);

        // Update data arrays for each lane
        for (int i = 0; i < NUM_LANES; i++) {
            data[3][i] = v1[i];
            data[4][i] = v2[i];
            data[5][i] = v3[i];
            data[6][i] = v4[i];
        }

        // Compute MD5 hashes for all lanes
        md5_cpu_avx2((v8si *)data, (v8si *)hash);

        // Process each lane separately
        for (int lane = 0; lane < NUM_LANES; lane++) {
            if (hash[3][lane]== 0) {
                for (int i = 0; i < 13; i++) {
                    coin[i] = data[i][lane];
                }
                save_deti_coin(coin);
                n_coins++;
            }
        }
    }

    STORE_DETI_COINS();
    printf("deti_coins_cpu_search: %lu DETI coin%s found in %lu attempt%s (expected %.2f coins)\n",n_coins,(n_coins == 1ul) ? "" : "s",n_attempts,(n_attempts == 1ul) ? "" : "s",(double)n_attempts / (double)(1ul << 32));
}

#endif