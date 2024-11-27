#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#include <opencl-c.h>
#include "md5.h"

typedef unsigned int uint;

__kernel void deti_coins_opencl_kernel_search(
    __global uint *deti_coins_storage_area,
    uint random_word,
    uint custom_word_1,
    uint custom_word_2)
{
    uint n = get_global_id(0);
    uint coin_counter = 0;

    uint coin[13];
    uint hash[4];

    // Inicializa o estado e buffers necessários para MD5
    uint state[4]; // Adicionado para evitar erros de 'state'
    uint x[16];    // Adicionado para evitar erros de 'x'
    uint a, b, c, d; 

    // Inicializa o coin template
    coin[0] = 0x49544544; // "DETI" em little-endian
    coin[1] = 0x696F6320; // "coi "
    coin[2] = 0x6E20206E; // "n  n"
    coin[3] = custom_word_1; // Valor customizado
    coin[4] = custom_word_2; // Valor customizado
    coin[5] = 0x20202020;
    coin[5] += (n % 64) << 0; 
    n /= 64;
    coin[5] += (n % 64) << 8;
    n /= 64;
    coin[5] += (n % 64) << 16;
    n /= 64;
    coin[5] += (n % 64) << 24;
    n /= 64;
    for (int i = 6; i < 12; i++) coin[i] = 0x20202020; // Padding
    coin[12] = 0x0A202020; // Ending

    for (int i = 0; i < 64; i++) {
        // Incrementa coin[7]
        coin[7]++;

        // Computa o hash MD5 (substitua pela implementação real)
        #define C(c)         (c)
        #define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
        #define DATA(idx)    coin[idx]
        #define HASH(idx)    hash[idx]
        #define STATE(idx)   state[idx]
        #define X(idx)       x[idx]
        CUSTOM_MD5_CODE(); // Substitua pela implementação real
        #undef C
        #undef ROTATE
        #undef DATA
        #undef HASH
        #undef STATE
        #undef X

        // Verifica o resultado do hash
        if (hash[3] == 0) {
            uint a = atomic_add(&deti_coins_storage_area[0], 13u);
            if (a <= 1024 - 13) {
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
