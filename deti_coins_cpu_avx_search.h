#ifndef DETI_COINS_CPU_AVX_SEARCH
#define DETI_COINS_CPU_AVX_SEARCH

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>  // Added for rand(), srand()
#include <time.h>    // Added for time()
#include <immintrin.h>
#include "md5_cpu_avx.h"
#include "cpu_utilities.h"

typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef uint8_t u08_t;

extern volatile int stop_request;

// Function to generate a random u32_t with printable ASCII characters
static u32_t random_printable_u32() {
    u32_t v = 0;
    for (int i = 0; i < 4; i++) {
        u08_t c = (rand() % (0x7E - 0x20 + 1)) + 0x20; // Random character between 0x20 and 0x7E
        v |= ((u32_t)c) << (i * 8);
    }
    return v;
}

static void deti_coins_cpu_avx_search(uint32_t n_random_words) {
    u32_t n;
    u64_t n_attempts = 0, n_coins = 0;
    u32_t data[13][4] __attribute__((aligned(32)));
    u32_t hash[4][4];
    u32_t v1[4], v2[4], v3[4], v4[4], v5[4], v6[4];


    // Seed the random number generator
    srand(time(NULL));

    // Initialize v1 and v2 for each lane with random values
    for (int i = 0; i < 4; i++) {
        v1[i] = random_printable_u32();
        v2[i] = random_printable_u32();
        v3[i] = random_printable_u32();
        v4[i] = random_printable_u32();
        v5[i] = random_printable_u32();
        v6[i] = random_printable_u32();
    }


    // Initialization of 'data' with static values
    // First, initialize all data to spaces
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 4; j++) {
            data[i][j] = 0x20202020; // Little-endian '    '
        }
    }

    // Set the specific initial values
    for (int j = 0; j < 4; j++) {
        data[0][j] = 0x49544544; // "DETI" in little-endian
        data[1][j] = 0x696F6320; // "coi " in little-endian
        data[2][j] = 0x6E20206E; // "n  n" in little-endian
        data[3][j] = v1[j];
        data[4][j] = v2[j];
        data[5][j] = v3[j];
        data[6][j] = v4[j];
        data[7][j] = v5[j];
        data[8][j] = v6[j];
        data[12][j] = 0x0A202020; // Newline character '\n' followed by spaces
    }

    u32_t coin[13];

    // Main loop to find DETI coins
    for (n_attempts = n_coins = 0ul; stop_request == 0; n_attempts += 4) {
        for (int i = 0; i < 4; i++) {
            v1[i] = random_printable_u32();
            v2[i] = random_printable_u32();
            
            

            // Update data arrays for each lane
            data[3][i] = v1[i];
            data[4][i] = v2[i];
        }

        // Compute MD5 hashes for all lanes
        md5_cpu_avx((v4si *)data, (v4si *)hash);

        // Process each lane separately
        for (int lane = 0; lane < 4; lane++) {
            u32_t hash_lane[4];
            // Extract the hash for the current lane
            for (int i = 0; i < 4; i++) {
                hash_lane[i] = hash[i][lane];
            }

            // Reverse bytes if necessary (depends on your `hash_byte_reverse` implementation)
            hash_byte_reverse(hash_lane);

            // Calculate the power of the coin
            n = deti_coin_power(hash_lane);

            if (n >= 30) {
                printf("Number of trailing zeros: %d\n", n);
                printf("Lane %d: %08X %08X %08X %08X\n", lane, hash_lane[0], hash_lane[1], hash_lane[2], hash_lane[3]);
            }

            // If the coin meets the criteria, save it
            if (n >= 32u) {
                // Save the coin data
                for (int i = 0; i < 13; i++) {
                    coin[i] = data[i][lane];
                }
                save_deti_coin(coin);
                n_coins++;
            }
        }
    }

    // Save and display the final count of DETI coins found
    STORE_DETI_COINS();
    printf("deti_coins_cpu_avx_search: %lu DETI coin%s found in %lu attempts\n",
           n_coins, n_coins == 1 ? "" : "s", n_attempts);
}

#endif
