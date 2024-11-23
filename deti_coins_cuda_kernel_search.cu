#include <md5.h>

typedef unsigned int u32_t;
typedef unsigned char u08_t;

__device__ u32_t inc(u32_t v) {
    v += 0x00000001;
    if ((v & 0x000000ff) != 0x0000007f) return v;
    v += 0x000000A1;
    if ((v & 0x000000ff) != 0x0000007f) return v;
    return v;
}

extern "C" __global__ __launch_bounds__(128, 1) void cuda_md5_kernel_search(
    u32_t *interleaved32_data,
    u32_t *interleaved32_hash,
    u32_t *deti_coins_storage_area,
    u32_t custom_word_1,
    u32_t custom_word_2,
    u32_t custom_word_3)
{
    u32_t n, a, b, c, d, state_local[4], x[16], hash[4];
    u32_t coin[13];

    // Get global thread ID
    n = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize DETI coin template
    coin[0] = 0x49544544; // "DETI" in little-endian
    coin[1] = 0x696F6320; // "coi "
    coin[2] = 0x6E20206E; // "n  n"
    coin[3] = custom_word_1;
    coin[4] = custom_word_2;
    coin[5] = custom_word_3;
    for (int i = 6; i < 12; i++) coin[i] = 0x20202020; // Padding
    coin[12] = 0x0A202020; // Ending

    for (int i = 0; i < 64; i++) {
        // Derive unique values for custom words based on thread ID and iteration
        coin[3] = inc(custom_word_1);
        if (coin[3] == 0x20202020) {
            coin[4] = inc(custom_word_2);
        }

        // Compute MD5 hash
        #define C(c)         (c)
        #define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
        #define DATA(idx)    interleaved32_data[13 * (idx)]
        #define HASH(idx)    interleaved32_hash[32 * (idx)]
        #define STATE(idx)   state_local[idx]
        #define X(idx)       x[idx]
        CUSTOM_MD5_CODE();
        #undef C
        #undef ROTATE
        #undef DATA
        #undef HASH
        #undef STATE
        #undef X

        // If hash meets criteria, store the coin
        if (hash[3] == 0) {
            a = atomicAdd(deti_coins_storage_area, 13u);
            if (a + 13 <= 1024 * 13) {
                for (int j = 0; j < 13; j++) {
                    deti_coins_storage_area[a + j] = coin[j];
                }
            }
        }

        // Increment combination
        interleaved32_data[13 * n]++;
    }
}