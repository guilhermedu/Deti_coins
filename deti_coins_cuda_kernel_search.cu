typedef unsigned int u32_t;
typedef unsigned char u08_t;

#include "md5.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>



__device__ unsigned int coin_counter = 0;


extern "C" __global__ __launch_bounds__(128, 1) void deti_coins_cuda_kernel_search(
    u32_t *deti_coins_storage_area,
    u32_t custom_word_1,
    u32_t custom_word_2)
{
    u32_t n, a, b, c, d, state[4], x[16], hash[4]; 
    u32_t coin[13];

    // Get global thread ID
    n = threadIdx.x + blockIdx.x * blockDim.x;
        
    // Initialize DETI coin template
    coin[0] = 0x49544544; // "DETI" in little-endian
    coin[1] = 0x696F6320; // "coi "
    coin[2] = 0x6E20206E; // "n  n"
    coin[3] = custom_word_1; // Inicializa com um valor único para cada thread
    coin[4] = custom_word_2; // Inicializa com um valor único para cada thread
    coin[5] = 0x20202020;
    coin[5] += (n % 64) << 16; 
    n/=64;
    coin[5] += (n % 64) << 24;
    n/=64;
    coin[5] += (n % 64) << 0;
    n/=64;
    coin[5] += (n % 64) << 8;
    n /= 64;
    for (int i = 6; i < 12; i++) coin[i] = 0x20202020; // Padding
    coin[12] = 0x0A202020; // Ending



    

    for ( n = 0; n < 64; n++) {
        coin[7]++;
        //printf("Coin:%08x\n", coin[3]);
        // Compute MD5 hash
        #define C(c)         (c)
        #define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
        #define DATA(idx)    coin[idx]
        #define HASH(idx)    hash[idx]
        #define STATE(idx)   state[idx]
        #define X(idx)       x[idx]
        CUSTOM_MD5_CODE();
        #undef C
        #undef ROTATE
        #undef DATA
        #undef HASH
        #undef STATE
        #undef X

        // If hash meets criteria, store the coin 
        if (hash[3] == 0 ) {    
            //printf("Found a coin!\n");
            a = atomicAdd(deti_coins_storage_area, 13u);
            if (a <= 1024 -13) {
                deti_coins_storage_area[a+0] = coin[0];
                deti_coins_storage_area[a+1] = coin[1];
                deti_coins_storage_area[a+2] = coin[2];
                deti_coins_storage_area[a+3] = coin[3];
                deti_coins_storage_area[a+4] = coin[4];
                deti_coins_storage_area[a+5] = coin[5];
                deti_coins_storage_area[a+6] = coin[6];
                deti_coins_storage_area[a+7] = coin[7];
                deti_coins_storage_area[a+8] = coin[8];
                deti_coins_storage_area[a+9] = coin[9];
                deti_coins_storage_area[a+10] = coin[10];
                deti_coins_storage_area[a+11] = coin[11];
                deti_coins_storage_area[a+12] = coin[12];
                
            }
        }
    }
}