//
// Tomás Oliveira e Silva,  October 2024
//
// Arquiteturas de Alto Desempenho 2024/2025
//
// MD5 hash CPU code
//
// md5_cpu() -------- compute the MD5 hash of a message
// test_md5_cpu() --- test the correctness of md5_cpu() and measure its execution time
//

#ifndef MD5_CPU
#define MD5_CPU

#include "md5.h"
#include <string.h>
//
// CPU-only implementation (assumes a little-endian CPU)
//

static void md5_cpu(u32_t *data,u32_t *hash)
{ // one message -> one MD5 hash
  u32_t a,b,c,d,state[4],x[16];
# define C(c)         (c)
# define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
# define DATA(idx)    data[idx]
# define HASH(idx)    hash[idx]
# define STATE(idx)   state[idx]
# define X(idx)       x[idx]
  CUSTOM_MD5_CODE();
# undef C
# undef ROTATE
# undef DATA
# undef HASH
# undef STATE
# undef X
}


//
// correctness test of md5_cpu()
//

static void test_md5_cpu(void)
{
# define N_MD5SUM_TESTS  64u
# define N_TIMING_TESTS  1000000u
  u32_t n,idx,*htd,*hth;


  htd = &host_md5_test_data[0u];
  hth = &host_md5_test_hash[0u];
  //
  // do all messages
  //
  for(n = 0u;n < N_MESSAGES;n++)
  {
    //
    // compute the MD5 hash
    //
    md5_cpu(htd,hth);
    //
    // compare the MD5 hash with the output of the md5sum command
    //
    if (n < N_MD5SUM_TESTS) {
    unsigned char computed_hash[16];

    // Calcula o hash MD5 dos dados htd
    MD5((unsigned char *)htd, 13 * 4, computed_hash);

    // Verifica se o hash calculado corresponde ao#include <string.h> esperado
    for (idx = 0; idx < 4; idx++) {
        // Converte os 4 bytes do hash para u32_t, considerando endianess
        u32_t hash_part;
        memcpy(&hash_part, &computed_hash[idx * 4], 4);
        hash_part = SWAP(hash_part);

        if (hash_part != hth[idx]) {
            fprintf(stderr, "test_md5_cpu: MD5 hash error for message %u\n", n);
            exit(1);
        }
    }
    }
    //
    // advance to the next message
    //
    htd = &htd[13u];
    hth = &hth[ 4u];
  }
  //
  // measure the execution time of mp5_cpu()
  //
# if N_TIMING_TESTS > 0u
  time_measurement();
  for(n = 0u;n < N_TIMING_TESTS;n++)
    md5_cpu(&host_md5_test_data[0u],&host_md5_test_hash[0u]);
  time_measurement();
  printf("time per md5 hash ( cpu): %7.3fns %7.3fns\n",cpu_time_delta_ns() / (double)N_TIMING_TESTS,wall_time_delta_ns() / (double)N_TIMING_TESTS);
# endif
# undef N_MD5SUM_TESTS
# undef N_TIMING_TESTS
}

#endif
