//
// Tomás Oliveira e Silva,  October 2024
//
// Arquiteturas de Alto Desempenho 2024/2025
//
// MD5 hash CPU code using AVX instructions (Intel/AMD)
//
// md5_cpu_avx() -------- compute the MD5 hash of a message
// test_md5_cpu_avx() --- test the correctness of md5_cpu() and measure its execution time
//

//#if defined(__GNUC__) && defined(__AVX__)
#ifndef MD5_CPU_AVX
#define MD5_CPU_AVX
// md5_cpu_avx.c
#include "debug.h"

typedef int v4si __attribute__ ((vector_size (16)));

static void md5_cpu_avx(v4si *interleaved4_data, v4si *interleaved4_hash) {
    v4si a, b, c, d, interleaved4_state[4], interleaved4_x[16];
# define C(c)         (v4si){ (int)(c), (int)(c), (int)(c), (int)(c) }
# define ROTATE(x,n)  (__builtin_ia32_pslldi128(x,n) | __builtin_ia32_psrldi128(x,32 - (n)))
# define DATA(idx)    interleaved4_data[idx]
# define HASH(idx)    interleaved4_hash[idx]
# define STATE(idx)   interleaved4_state[idx]
# define X(idx)       interleaved4_x[idx]
CUSTOM_MD5_CODE();

    /* // Print initial data for each lane
    if (debug_active) {
        printf("md5_cpu_avx initial data:\n");
        for (int i = 0; i < 16; i++) {
            printf("DATA[%d] = { %08X, %08X, %08X, %08X }\n", i,
                   DATA(i)[0], DATA(i)[1], DATA(i)[2], DATA(i)[3]);
        }

        // Print initial state for each lane
        printf("md5_cpu_avx initial state:\n");
        for (int i = 0; i < 4; i++) {
            printf("STATE[%d] = { %08X, %08X, %08X, %08X }\n", i,
                   STATE(i)[0], STATE(i)[1], STATE(i)[2], STATE(i)[3]);
        }
    }

    CUSTOM_MD5_CODE();

    // Print final hash for each lane
    if (debug_active) {
        printf("md5_cpu_avx final hash:\n");
        for (int i = 0; i < 4; i++) {
            printf("HASH[%d] = { %08X, %08X, %08X, %08X }\n", i,
                   HASH(i)[0], HASH(i)[1], HASH(i)[2], HASH(i)[3]);
        }
    }

    debug_active = 0; // Clear debug flag */
# undef C
# undef ROTATE
# undef DATA
# undef HASH
# undef STATE
# undef X
}


//
// correctness test of md5_cpu_avx() --- test_md5_cpu() must be called first!
//

static void test_md5_cpu_avx(void)
{
# define N_TIMING_TESTS  1000000u
  static u32_t interleaved_test_data[13u * 4u] __attribute__((aligned(16)));
  static u32_t interleaved_test_hash[ 4u * 4u] __attribute__((aligned(16)));
  u32_t n,lane,idx,*htd,*hth;

  if(N_MESSAGES % 4u != 0u)
  {
    fprintf(stderr,"test_md5_cpu_avx: N_MESSAGES is not a multiple of 4\n");
    exit(1);
  }
  htd = &host_md5_test_data[0u];
  hth = &host_md5_test_hash[0u];
  for(n = 0u;n < N_MESSAGES;n += 4u)
  {
    //
    // interleave data
    //
    for(lane = 0u;lane < 4u;lane++)                                      // for each message number
      for(idx = 0u;idx < 13u;idx++)                                      //  for each message word
        interleaved_test_data[4u * idx + lane] = htd[13u * lane + idx];  //   interleave
    //
    // compute MD5 hashes
    //
    md5_cpu_avx((v4si *)interleaved_test_data,(v4si *)interleaved_test_hash);
    //
    // compare with the test_md5_cpu() data
    //
    for(lane = 0u;lane < 4u;lane++)  // for each message number
      for(idx = 0u;idx < 4u;idx++)   //  for each hash word
        if(interleaved_test_hash[4u * idx + lane] != hth[4u * lane + idx])
        {
          fprintf(stderr,"test_md5_cpu_avx: MD5 hash error for message %u\n",4u * n + lane);
          exit(1);
        }
    //
    // advance to the next 4 messages
    //
    htd = &htd[13u * 4u];
    hth = &hth[ 4u * 4u];
  }
  //
  // measure the execution time of mp5_cpu_avx()
  //
# if N_TIMING_TESTS > 0u
  time_measurement();
  for(n = 0u;n < N_TIMING_TESTS;n++)
    md5_cpu_avx((v4si *)interleaved_test_data,(v4si *)interleaved_test_hash);
  time_measurement();
  printf("time per md5 hash ( avx): %7.3fns %7.3fns\n",cpu_time_delta_ns() / (double)(4u * N_TIMING_TESTS),wall_time_delta_ns() / (double)(4u * N_TIMING_TESTS));
# endif
# undef N_TIMING_TESTS
}

#endif
//#endif
